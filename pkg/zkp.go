package pkg

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/frontend"
	"gnarktest/config"
	"gnarktest/model"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// 为每个向量长度维护独立的电路和密钥
type Groth16Manager struct {
	circuits      map[int]constraint.ConstraintSystem
	provingKeys   map[int]groth16.ProvingKey
	verifyingKeys map[int]groth16.VerifyingKey
	mutex         sync.RWMutex
}

var groth16Manager = &Groth16Manager{
	circuits:      make(map[int]constraint.ConstraintSystem),
	provingKeys:   make(map[int]groth16.ProvingKey),
	verifyingKeys: make(map[int]groth16.VerifyingKey),
}

// 初始化指定向量长度的电路和密钥
func (gm *Groth16Manager) InitForLength(length int) error {
	gm.mutex.Lock()
	defer gm.mutex.Unlock()

	// 检查是否已经初始化
	if _, exists := gm.circuits[length]; exists {
		return nil
	}

	// 构建路径
	basePath := filepath.Join(config.BasePath, config.GrothPath, fmt.Sprintf("length_%d", length))
	ccsPath := filepath.Join(basePath, config.GrothCcsPath)
	pkPath := filepath.Join(basePath, config.GrothPkPath)
	vkPath := filepath.Join(basePath, config.GrothVkPath)

	// 尝试读取已存在的电路和密钥
	if ccs, err := ReadCcs(ccsPath); err == nil {
		if pk, err := ReadPk(pkPath); err == nil {
			if vk, err := ReadVk(vkPath); err == nil {
				gm.circuits[length] = ccs
				gm.provingKeys[length] = pk
				gm.verifyingKeys[length] = vk
				return nil
			}
		}
	}

	// 如果文件不存在，需要重新生成
	return fmt.Errorf("circuit and keys for length %d not found, please generate them first", length)
}

// 获取指定长度的电路
func (gm *Groth16Manager) GetCircuit(length int) (constraint.ConstraintSystem, error) {
	gm.mutex.RLock()
	defer gm.mutex.RUnlock()

	if ccs, exists := gm.circuits[length]; exists {
		return ccs, nil
	}
	return nil, fmt.Errorf("circuit for length %d not initialized", length)
}

// 获取指定长度的证明密钥
func (gm *Groth16Manager) GetProvingKey(length int) (groth16.ProvingKey, error) {
	gm.mutex.RLock()
	defer gm.mutex.RUnlock()

	if pk, exists := gm.provingKeys[length]; exists {
		return pk, nil
	}
	return nil, fmt.Errorf("proving key for length %d not initialized", length)
}

// 获取指定长度的验证密钥
func (gm *Groth16Manager) GetVerifyingKey(length int) (groth16.VerifyingKey, error) {
	gm.mutex.RLock()
	defer gm.mutex.RUnlock()

	if vk, exists := gm.verifyingKeys[length]; exists {
		return vk, nil
	}
	return nil, fmt.Errorf("verifying key for length %d not initialized", length)
}

// 初始化所有支持的向量长度
func InitGroth() {
	for _, length := range config.GlobalVectorManager.GetSupportedLengths() {
		if err := groth16Manager.InitForLength(length); err != nil {
			fmt.Printf("Warning: Failed to initialize Groth16 for length %d: %v\n", length, err)
		}
	}
}

// 随机生成Q，K，L
func GenRandVector(len int) ([]int64, []int64, int64) {
	rand.Seed(time.Now().UnixNano())

	a := make([]int64, len)
	b := make([]int64, len)
	for i := 0; i < len; i++ {
		a[i] = int64(rand.Intn(10000) - 5000)
		b[i] = int64(rand.Intn(10000) - 5000)
	}
	innerProduct := CalculateInnerProduct(a, b)
	return a, b, innerProduct
}

func CalculateInnerProduct(a, b []int64) int64 {
	innerProduct := int64(0)
	for i := 0; i < len(a); i++ {
		innerProduct += a[i] * b[i]
	}
	return innerProduct
}

// Handler调用 - 使用当前向量长度
func GenGrothProof(q []int64, k []int64) (string, *model.Cost, error) {
	currentLength := config.GlobalVectorManager.GetCurrentLength()
	return GenGrothProofWithLength(q, k, currentLength)
}

// 指定向量长度的证明生成
func GenGrothProofWithLength(q []int64, k []int64, length int) (string, *model.Cost, error) {
	// 检查向量长度是否匹配
	if len(q) != length || len(k) != length {
		return "", nil, fmt.Errorf("vector length mismatch: expected %d, got q=%d, k=%d", length, len(q), len(k))
	}

	// 获取对应的电路和密钥
	ccs, err := groth16Manager.GetCircuit(length)
	if err != nil {
		return "", nil, err
	}

	pk, err := groth16Manager.GetProvingKey(length)
	if err != nil {
		return "", nil, err
	}

	assignment := model.InnerProductCircuit{
		A:       ConvertToElement(q),
		B:       ConvertToElement(k),
		Desired: CalculateInnerProduct(q, k),
	}
	cost := new(model.Cost)
	start := time.Now()
	witness, err := frontend.NewWitness(&assignment, ecc.BN254.ScalarField())
	if err != nil {
		return "", nil, err
	}
	proof, err := groth16.Prove(ccs, pk, witness)
	{
		elapsed := time.Since(start)
		cost.Time = elapsed
		var buff bytes.Buffer
		proof.WriteTo(io.Writer(&buff))
		cost.Space = int64(buff.Len())
	}
	if err != nil {
		return "", nil, err
	}
	return ConvertProof2String(proof), cost, nil
}

// Handler调用 - 使用当前向量长度
func VerifyGrothProof(proofStr string, q []int64, l int64) (bool, *model.Cost, error) {
	currentLength := config.GlobalVectorManager.GetCurrentLength()
	return VerifyGrothProofWithLength(proofStr, q, l, currentLength)
}

// 指定向量长度的证明验证
func VerifyGrothProofWithLength(proofStr string, q []int64, l int64, length int) (bool, *model.Cost, error) {
	// 检查向量长度是否匹配
	if len(q) != length {
		return false, nil, fmt.Errorf("vector length mismatch: expected %d, got %d", length, len(q))
	}

	// 获取对应的验证密钥
	vk, err := groth16Manager.GetVerifyingKey(length)
	if err != nil {
		return false, nil, err
	}

	publicInput := model.InnerProductCircuit{
		A:       ConvertToElement(q),
		Desired: l,
	}
	publicWitness, err := frontend.NewWitness(&publicInput, ecc.BN254.ScalarField(), frontend.PublicOnly())
	if err != nil {
		return false, nil, err
	}
	proof, err := ConvertString2Proof(proofStr)
	if err != nil {
		return false, nil, err
	}
	cost := new(model.Cost)
	start := time.Now()
	err = groth16.Verify(proof, vk, publicWitness)
	elapsed := time.Since(start)
	cost.Time = elapsed
	if err != nil {
		return false, nil, err
	}
	return true, cost, nil
}

func ConvertProof2String(proof groth16.Proof) string {
	var buff bytes.Buffer
	proof.WriteTo(&buff)
	return base64.StdEncoding.EncodeToString(buff.Bytes())
}

func ConvertString2Proof(s string) (groth16.Proof, error) {
	proof := groth16.NewProof(ecc.BN254)

	decodedBytes, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		return proof, err
	}
	var buff bytes.Buffer
	buff.Write(decodedBytes)
	_, err = proof.ReadFrom(&buff)
	if err != nil {
		return proof, err
	}
	return proof, nil
}

func ConvertToElement(a []int64) []frontend.Variable {
	var res []frontend.Variable
	for i := 0; i < len(a); i++ {
		res = append(res, a[i])
	}
	return res
}

func SaveCcs(ccs constraint.ConstraintSystem) error {
	if err := os.MkdirAll(filepath.Join(config.BasePath, config.GrothPath), 0755); err != nil {
		fmt.Printf("目录%s已存在\n", filepath.Join(config.BasePath, config.GrothPath))
	}
	ccsFile, err := os.Create(filepath.Join(config.BasePath, config.GrothPath, config.GrothCcsPath))
	if err != nil {
		return fmt.Errorf("创建 cs 失败: %w", err)
	}
	defer ccsFile.Close()

	if _, err = ccs.WriteTo(ccsFile); err != nil {
		return fmt.Errorf("写入 cs 失败: %w", err)
	}
	return nil
}

func ReadCcs(path string) (constraint.ConstraintSystem, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	ccs := groth16.NewCS(ecc.BN254)
	if _, err = ccs.ReadFrom(file); err != nil {
		return nil, err
	}
	return ccs, nil
}

func SaveGrothKeys(pk groth16.ProvingKey, vk groth16.VerifyingKey) error {
	if err := os.MkdirAll(filepath.Join(config.BasePath, config.GrothPath), 0755); err != nil {
		fmt.Printf("目录%s已存在\n", filepath.Join(config.BasePath, config.GrothPath))
	}
	// 保存 Proving Key
	pkFile, err := os.Create(filepath.Join(config.BasePath, config.GrothPath, config.GrothPkPath))
	if err != nil {
		return fmt.Errorf("创建 proving.key 失败: %w", err)
	}
	defer pkFile.Close()

	if _, err = pk.WriteTo(pkFile); err != nil {
		return fmt.Errorf("写入 proving.key 失败: %w", err)
	}

	// 保存 Verifying Key
	vkFile, err := os.Create(filepath.Join(config.BasePath, config.GrothPath, config.GrothVkPath))
	if err != nil {
		return fmt.Errorf("创建 verifying.key 失败: %w", err)
	}
	defer vkFile.Close()

	if _, err := vk.WriteTo(vkFile); err != nil {
		return fmt.Errorf("写入 verifying.key 失败: %w", err)
	}
	return nil
}

func ReadVk(path string) (groth16.VerifyingKey, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	vk := groth16.NewVerifyingKey(ecc.BN254)
	if _, err = vk.ReadFrom(file); err != nil {
		return nil, err
	}
	return vk, nil
}
func ReadPk(path string) (groth16.ProvingKey, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	pk := groth16.NewProvingKey(ecc.BN254)
	if _, err = pk.ReadFrom(file); err != nil {
		return nil, err
	}
	return pk, nil
}

func SaveProof(proof groth16.Proof, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("创建 verifying.key 失败: %w", err)
	}
	defer file.Close()

	if _, err := proof.WriteTo(file); err != nil {
		return fmt.Errorf("写入 verifying.key 失败: %w", err)
	}
	return nil
}

func ReadProof(path string) (groth16.Proof, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	proof := groth16.NewProof(ecc.BN254)
	if _, err = proof.ReadFrom(file); err != nil {
		return nil, err
	}
	return proof, nil
}

// 获取Groth16Manager实例
func GetGroth16Manager() *Groth16Manager {
	return groth16Manager
}
