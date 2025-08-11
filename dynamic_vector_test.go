package main

import (
	"fmt"
	"gnarktest/config"
	"gnarktest/pkg"
	"testing"
)

// 测试动态向量长度切换
func TestDynamicVectorLength(t *testing.T) {
	// 初始化
	pkg.InitGroth()
	
	// 测试不同的向量长度
	testLengths := []int{128, 256, 384}
	
	for _, length := range testLengths {
		t.Run(fmt.Sprintf("Length_%d", length), func(t *testing.T) {
			// 设置当前向量长度
			if !config.GlobalVectorManager.SetVectorLength(length) {
				t.Fatalf("Failed to set vector length to %d", length)
			}
			
			// 生成随机向量
			q, k, l := pkg.GenRandVector(length)
			
			// 生成证明
			proof, genCost, err := pkg.GenGrothProofWithLength(q, k, length)
			if err != nil {
				t.Fatalf("Failed to generate proof for length %d: %v", length, err)
			}
			
			// 验证证明
			valid, verifyCost, err := pkg.VerifyGrothProofWithLength(proof, q, l, length)
			if err != nil {
				t.Fatalf("Failed to verify proof for length %d: %v", length, err)
			}
			
			if !valid {
				t.Fatalf("Proof verification failed for length %d", length)
			}
			
			fmt.Printf("Length %d: Generation time: %v, Verification time: %v\n", 
				length, genCost.Time, verifyCost.Time)
		})
	}
}

// 测试向量长度不匹配的情况
func TestVectorLengthMismatch(t *testing.T) {
	pkg.InitGroth()
	
	// 设置向量长度为256
	config.GlobalVectorManager.SetVectorLength(256)
	
	// 尝试使用128长度的向量
	q, k, _ := pkg.GenRandVector(128)
	
	_, _, err := pkg.GenGrothProofWithLength(q, k, 256)
	if err == nil {
		t.Fatal("Expected error for vector length mismatch, but got none")
	}
	
	fmt.Printf("Expected error caught: %v\n", err)
}

// 测试生成电路和密钥
func TestGenerateCircuits(t *testing.T) {
	// 为所有支持的向量长度生成电路和密钥
	err := pkg.GenerateAllCircuitsAndKeys()
	if err != nil {
		t.Fatalf("Failed to generate circuits and keys: %v", err)
	}
	
	fmt.Println("All circuits and keys generated successfully")
}

// 测试获取当前向量长度
func TestGetCurrentVectorLength(t *testing.T) {
	currentLength := config.GlobalVectorManager.GetCurrentLength()
	supportedLengths := config.GlobalVectorManager.GetSupportedLengths()
	
	fmt.Printf("Current vector length: %d\n", currentLength)
	fmt.Printf("Supported lengths: %v\n", supportedLengths)
	
	// 验证当前长度在支持列表中
	found := false
	for _, length := range supportedLengths {
		if length == currentLength {
			found = true
			break
		}
	}
	
	if !found {
		t.Fatalf("Current length %d is not in supported lengths %v", currentLength, supportedLengths)
	}
} 