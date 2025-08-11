package handler

import (
	"crypto/sha256"
	"encoding/base64"
	"github.com/gin-gonic/gin"
	"gnarktest/model"
	"gnarktest/pkg"
	"time"
)

func VerifyPederSonAndMerkleHandler() func(c *gin.Context) {
	return func(c *gin.Context) {
		var req model.VerifyPoRReq
		c.BindJSON(&req)
		var cp model.CommitAndMerkleProof
		cp.LoadFromFile(req.ProofId)
		cost1, res := pkg.VerifyMerkleProofByString(cp.Data, cp.MerkleProof.Proof, cp.MerkleProof.Index, cp.MerkleProof.Root, sha256.New)
		if !res {
			c.JSON(200, gin.H{"code": 405, "msg": "merkle proof verify failed"})
			return
		}
		var err error
		cost2 := new(model.Cost)
		res, cost2, err = pkg.VerifyCommitment(cp.Commitment.Commits, pkg.TransFloat64ToInt64(cp.Q), cp.L, cp.Commitment.R)
		if err != nil {
			c.JSON(200, gin.H{"code": 403, "msg": err.Error()})
			return
		}
		if !res {
			c.JSON(200, gin.H{"code": 405, "msg": "commitment verify failed"})
			return
		}
		c.JSON(200, gin.H{"code": 200, "msg": "ok", "time_cost": cost1.Time + cost2.Time})
	}
}

func VerifyGrothAndMerkleHandler() func(c *gin.Context) {
	return func(c *gin.Context) {
		var req model.VerifyPoRReq
		c.BindJSON(&req)
		var gp model.GrothAndMerkleProof
		gp.LoadFromFile(req.ProofId)
		cost1, res := pkg.VerifyMerkleProofByString(gp.Data, gp.MerkleProof.Proof, gp.MerkleProof.Index, gp.MerkleProof.Root, sha256.New)
		if !res {
			c.JSON(200, gin.H{"code": 405, "msg": "merkle proof verify failed"})
			return
		}
		var err error
		var cost2 *model.Cost
		res, cost2, err = pkg.VerifyGrothProof(gp.GrothProof, pkg.TransFloat64ToInt64(gp.Q), gp.L)
		if err != nil {
			c.JSON(200, gin.H{"code": 403, "msg": err.Error()})
			return
		}
		if !res {
			c.JSON(200, gin.H{"code": 405, "msg": "groth proof verify failed"})
			return
		}
		c.JSON(200, gin.H{"code": 200, "msg": "ok", "time_cost": cost1.Time + cost2.Time})
	}
}

func VerifyPoGHandler() func(c *gin.Context) {
	return func(c *gin.Context) {
		var req model.VerifyPoGReq
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(200, gin.H{"code": 405, "msg": err.Error()})
			return
		}
		var pog model.PoG
		pog.ReadFromFile(req.PoGId)
		var cpor model.CommitAndMerkleProof
		flag := false
		if err := cpor.LoadFromFile(req.PoRId); err == nil {
			flag = true
		}
		var gpor model.GrothAndMerkleProof
		if err := gpor.LoadFromFile(req.PoRId); err == nil {
			flag = false
		}
		var por model.PoR
		if flag {
			por = &cpor
		} else {
			por = &gpor
		}
		start := time.Now()
		res := false
		for i := 0; i < 1000; i++ {
			hash := sha256.Sum256([]byte(por.GetBaseProof().Data))
			hashSlice := hash[:]
			res = pog.Distance == por.GetBaseProof().L && pog.DataHash == base64.StdEncoding.EncodeToString(hashSlice)
		}
		elapsed := time.Since(start)
		if res {
			c.JSON(200, gin.H{"code": 200, "msg": "ok", "time_cost": elapsed / 1000})
		} else {
			c.JSON(200, gin.H{"code": 405, "msg": "verify failed"})
		}
	}
}
