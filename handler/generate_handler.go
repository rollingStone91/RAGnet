package handler

import (
	"crypto/sha256"
	"encoding/base64"
	"github.com/gin-gonic/gin"
	"gnarktest/model"
	"gnarktest/pkg"
	"gnarktest/utils"
	"time"
)

func GenPederSonAndMerkleHandler() func(c *gin.Context) {
	return func(c *gin.Context) {
		var req model.GenPoRReq
		c.ShouldBind(&req)
		commits, r, cost1, err := pkg.GenCommitment(pkg.TransFloat64ToInt64(req.K))
		if err != nil {
			c.JSON(200, gin.H{"code": 402, "msg": err.Error()})
			return
		}

		merkleProof, index, root, cost2, err := pkg.GenMerkleProof(req.ClientId, req.Data)
		if err != nil {
			c.JSON(200, gin.H{"code": 402, "msg": err.Error()})
			return
		}
		id, _ := utils.GenerateID()
		resp := model.CommitAndMerkleProof{
			BaseProof: model.BaseProof{
				ID:          id,
				MerkleProof: model.MerkleProof{merkleProof, index, root},
				Data:        req.Data,
				Q:           req.Q,
				L:           pkg.CalculateInnerProduct(pkg.TransFloat64ToInt64(req.Q), pkg.TransFloat64ToInt64(req.K)),
			},
			Commitment: model.Commitment{commits, r},
		}
		resp.Save2File()
		c.JSON(200, gin.H{"code": 200, "proof_id": id, "time_cost": cost1.Time + cost2.Time, "space_cost": cost1.Space + cost2.Space})
	}
}

func GenGrothAndMerkleHandler() func(c *gin.Context) {
	return func(c *gin.Context) {
		var req model.GenPoRReq
		c.ShouldBind(&req)
		proof, cost1, err := pkg.GenGrothProof(pkg.TransFloat64ToInt64(req.Q), pkg.TransFloat64ToInt64(req.K))
		if err != nil {
			c.JSON(200, gin.H{"code": 402, "msg": err.Error()})
			return
		}
		merkleProof, index, root, cost2, err := pkg.GenMerkleProof(req.ClientId, req.Data)
		if err != nil {
			c.JSON(200, gin.H{"code": 402, "msg": err.Error()})
			return
		}
		id, _ := utils.GenerateID()
		resp := model.GrothAndMerkleProof{
			BaseProof: model.BaseProof{
				ID:          id,
				MerkleProof: model.MerkleProof{merkleProof, index, root},
				Data:        req.Data,
				Q:           req.Q,
				L:           pkg.CalculateInnerProduct(pkg.TransFloat64ToInt64(req.Q), pkg.TransFloat64ToInt64(req.K)),
			},
			GrothProof: proof,
		}
		resp.Save2File()
		c.JSON(200, gin.H{"code": 200, "proof_id": id, "time_cost": cost1.Time + cost2.Time, "space_cost": cost1.Space + cost2.Space})
	}
}

func GenPoGHandler() func(c *gin.Context) {
	return func(c *gin.Context) {
		var req model.GenPoGReq
		if err := c.ShouldBind(&req); err != nil {
			c.JSON(200, gin.H{"code": 402, "msg": err.Error()})
			return
		}
		start := time.Now()
		var dist int64
		var hashSlice []byte
		for i := 0; i < 1000; i++ {
			dist = pkg.CalculateInnerProduct(pkg.TransFloat64ToInt64(req.Q), pkg.TransFloat64ToInt64(req.K))
			hash := sha256.Sum256([]byte(req.Data))
			hashSlice = hash[:] // 将数组转换为切片
		}
		elapsed := time.Since(start)
		id, _ := utils.GenerateID()
		resp := model.PoG{id, dist, base64.StdEncoding.EncodeToString(hashSlice)}
		resp.Save2File()
		c.JSON(200, gin.H{"code": 200, "proof_id": id, "time_cost": elapsed / 1000, "space_cost": len(hashSlice) + 8})
	}
}
