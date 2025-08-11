package main

import (
	"github.com/gin-gonic/gin"
	"gnarktest/handler"
	"gnarktest/pkg"
)

func main() {
	pkg.InitGroth()
	pkg.InitMerkleTrees()
	r := gin.Default()
	
	// 原有的路由
	r.POST("/gen_pedersen_merkle_proof", handler.GenPederSonAndMerkleHandler())
	r.POST("/gen_groth_merkle_proof", handler.GenGrothAndMerkleHandler())
	r.POST("/verify_pedersen_merkle_proof", handler.VerifyPederSonAndMerkleHandler())
	r.POST("/verify_groth_merkle_proof", handler.VerifyGrothAndMerkleHandler())
	r.POST("/gen_PoG", handler.GenPoGHandler())
	r.POST("/verify_PoG", handler.VerifyPoGHandler())
	
	// 新增的向量长度管理路由
	r.GET("/vector_length", handler.GetCurrentVectorLengthHandler())
	r.POST("/vector_length", handler.SetVectorLengthHandler())
	r.POST("/generate_circuit", handler.GenerateCircuitHandler())
	r.POST("/generate_all_circuits", handler.GenerateAllCircuitsHandler())
	r.GET("/test_vector_length", handler.TestVectorLengthHandler())
	
	r.Run(":8080")
}
