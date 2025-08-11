package pkg

import (
	"fmt"
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"gnarktest/config"
	"gnarktest/model"
	"os"
	"path/filepath"
)

// 为指定向量长度生成电路和密钥
func GenerateCircuitAndKeys(length int) error {
	// 检查是否支持该长度
	if !config.GlobalVectorManager.IsSupported(length) {
		return fmt.Errorf("unsupported vector length: %d", length)
	}
	
	// 创建目录
	basePath := filepath.Join(config.BasePath, config.GrothPath, fmt.Sprintf("length_%d", length))
	if err := os.MkdirAll(basePath, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}
	
	// 创建指定长度的电路
	circuit := createCircuitWithLength(length)
	
	// 编译电路
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &circuit)
	if err != nil {
		return fmt.Errorf("failed to compile circuit: %w", err)
	}
	
	// 生成密钥
	pk, vk, err := groth16.Setup(ccs)
	if err != nil {
		return fmt.Errorf("failed to setup keys: %w", err)
	}
	
	// 保存电路
	ccsPath := filepath.Join(basePath, config.GrothCcsPath)
	if err := SaveCcsToPath(ccs, ccsPath); err != nil {
		return fmt.Errorf("failed to save circuit: %w", err)
	}
	
	// 保存密钥
	pkPath := filepath.Join(basePath, config.GrothPkPath)
	vkPath := filepath.Join(basePath, config.GrothVkPath)
	if err := SaveGrothKeysToPath(pk, vk, pkPath, vkPath); err != nil {
		return fmt.Errorf("failed to save keys: %w", err)
	}
	
	fmt.Printf("Successfully generated circuit and keys for length %d\n", length)
	return nil
}

// 为所有支持的向量长度生成电路和密钥
func GenerateAllCircuitsAndKeys() error {
	for _, length := range config.GlobalVectorManager.GetSupportedLengths() {
		fmt.Printf("Generating circuit and keys for length %d...\n", length)
		if err := GenerateCircuitAndKeys(length); err != nil {
			fmt.Printf("Failed to generate for length %d: %v\n", length, err)
			return err
		}
	}
	fmt.Println("All circuits and keys generated successfully")
	return nil
}

// 创建指定长度的电路
func createCircuitWithLength(length int) model.InnerProductCircuit {
	// 创建指定长度的向量
	a := make([]frontend.Variable, length)
	b := make([]frontend.Variable, length)
	
	// 初始化向量（这里用0填充，实际使用时会被替换）
	for i := 0; i < length; i++ {
		a[i] = 0
		b[i] = 0
	}
	
	return model.InnerProductCircuit{
		A:       a,
		B:       b,
		Desired: 0,
	}
}

// 保存电路到指定路径
func SaveCcsToPath(ccs constraint.ConstraintSystem, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create circuit file: %w", err)
	}
	defer file.Close()

	if _, err = ccs.WriteTo(file); err != nil {
		return fmt.Errorf("failed to write circuit: %w", err)
	}
	return nil
}

// 保存密钥到指定路径
func SaveGrothKeysToPath(pk groth16.ProvingKey, vk groth16.VerifyingKey, pkPath, vkPath string) error {
	// 保存 Proving Key
	pkFile, err := os.Create(pkPath)
	if err != nil {
		return fmt.Errorf("failed to create proving key file: %w", err)
	}
	defer pkFile.Close()

	if _, err = pk.WriteTo(pkFile); err != nil {
		return fmt.Errorf("failed to write proving key: %w", err)
	}

	// 保存 Verifying Key
	vkFile, err := os.Create(vkPath)
	if err != nil {
		return fmt.Errorf("failed to create verifying key file: %w", err)
	}
	defer vkFile.Close()

	if _, err := vk.WriteTo(vkFile); err != nil {
		return fmt.Errorf("failed to write verifying key: %w", err)
	}
	return nil
} 