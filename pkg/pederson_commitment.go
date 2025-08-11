package pkg

import (
	"encoding/base64"
	"fmt"
	"github.com/bwesterb/go-ristretto"
	"gnarktest/config"
	"gnarktest/model"
	"math/big"
	"os"
	"path/filepath"
	"time"
)

func GetGenerateElement() (string, error) {
	data, err := os.ReadFile(filepath.Join(config.BasePath, config.PedersenPath))
	if err != nil {
		return "", err
	}
	// 转换为字符串
	content := string(data)
	return content, nil
}

func ParamsGen() (G ristretto.Point) {
	G.Rand()
	return G
}
func ParamsGenToString() (GString string, err error) {
	var G ristretto.Point
	G.Rand()
	GBytes, err := G.MarshalText()
	if err != nil {
		return "", err
	}
	GString = base64.StdEncoding.EncodeToString(GBytes)
	return
}

// 生成一个随机阶数
func RandomGen() (r ristretto.Scalar) {
	r.Rand()
	return r
}

func RandomGenToNumberString() string {
	var r ristretto.Scalar
	r.Rand()
	return r.BigInt().String()
}

func Commit(G ristretto.Point, secrets []int64, r ristretto.Scalar) (commits []ristretto.Point, cost *model.Cost, err error) {
	start := time.Now()
	scalars, _ := Int64sToScalars(secrets)

	var gr ristretto.Point
	gr.ScalarMult(&G, &r)

	for i := 0; i < len(scalars); i++ {
		var temp ristretto.Point
		temp.ScalarMult(&gr, &scalars[i])
		commits = append(commits, temp)
	}
	elapsed := time.Since(start)
	cost = new(model.Cost)
	cost.Time = elapsed
	return commits, cost, nil
}

func CommitFromString(GString string, k []int64, rString string) (commitStrings []string, cost *model.Cost, err error) {
	// 解析 H
	var G ristretto.Point
	GBytes, _ := base64.StdEncoding.DecodeString(GString)
	err = G.UnmarshalText(GBytes)
	if err != nil {
		return nil, nil, err
	}

	// 转换 r
	var r ristretto.Scalar
	var bigInt big.Int
	_, ok := bigInt.SetString(rString, 10)
	if !ok {
		return nil, nil, fmt.Errorf("invalid rString")
	}
	r.SetBigInt(&bigInt)

	// 计算承诺
	commits, cost, err := Commit(G, k, r)
	if err != nil {
		return nil, nil, err
	}
	sum := int64(0)
	for _, commit := range commits {
		bytes, err := commit.MarshalText()
		if err != nil {
			return nil, nil, err
		}
		commitString := base64.StdEncoding.EncodeToString(bytes)
		commitStrings = append(commitStrings, commitString)
		sum += int64(len(bytes))
	}
	cost.Space += sum
	if bytes, err := r.MarshalText(); err == nil {
		cost.Space += int64(len(bytes))
	} else {
		return nil, nil, err
	}

	return commitStrings, cost, nil
}

func Check(coms []ristretto.Point, G ristretto.Point, r ristretto.Scalar, q []int64, l int64) (bool, *model.Cost) {
	scalars, _ := Int64sToScalars(q)
	var com1 ristretto.Point
	start := time.Now()
	for i, com := range coms {
		//coms未初始化时不能直接相加
		if i == 0 {
			com1.ScalarMult(&com, &scalars[i])
		} else {
			com1.Add(&com1, com.ScalarMult(&com, &scalars[i]))
		}
	}
	var gr ristretto.Point
	gr.ScalarMult(&G, &r)
	var calcComm ristretto.Point
	sc, _ := Int64ToScalar(l)
	calcComm.ScalarMult(&gr, &sc)
	res := calcComm.Equals(&com1)
	elapsed := time.Since(start)
	cost := &model.Cost{elapsed, 0}
	return res, cost
}

func CheckByString(commitStrings []string, GString string, q []int64, rString string, l int64) (bool, *model.Cost, error) {
	// 解析 comm
	comms := make([]ristretto.Point, len(commitStrings))
	for i, commitString := range commitStrings {
		commBytes, err := base64.StdEncoding.DecodeString(commitString)
		if err != nil {
			return false, nil, err
		}
		err = comms[i].UnmarshalText(commBytes)
		if err != nil {
			return false, nil, err
		}
	}
	var g ristretto.Point
	gBytes, err := base64.StdEncoding.DecodeString(GString)
	if err != nil {
		return false, nil, err
	}
	err = g.UnmarshalText(gBytes)
	if err != nil {
		return false, nil, err
	}
	// 转换 r
	var r ristretto.Scalar
	var bigInt big.Int
	_, ok := bigInt.SetString(rString, 10)
	if !ok {
		return false, nil, fmt.Errorf("invalid rString")
	}
	r.SetBigInt(&bigInt)

	res, cost := Check(comms, g, r, q, l)
	return res, cost, nil
}

func Int64sToScalars(values []int64) ([]ristretto.Scalar, error) {
	scalars := make([]ristretto.Scalar, len(values))
	for i, v := range values {
		var bi big.Int
		bi.SetInt64(v)
		scalars[i].SetBigInt(&bi)
	}
	return scalars, nil
}

func Int64ToScalar(value int64) (ristretto.Scalar, error) {
	var bi big.Int
	bi.SetInt64(value)
	var scalar ristretto.Scalar
	scalar.SetBigInt(&bi)
	return scalar, nil
}

// 由handler调用
func GenCommitment(k []int64) (commits []string, r string, cost *model.Cost, err error) {
	gStr, _ := GetGenerateElement()
	r = RandomGenToNumberString()
	commits, cost, err = CommitFromString(gStr, k, r)
	if err != nil {
		return nil, "", nil, err
	}
	return commits, r, cost, nil
}

func VerifyCommitment(commitStrings []string, q []int64, l int64, rStr string) (bool, *model.Cost, error) {
	gStr, _ := GetGenerateElement()
	return CheckByString(commitStrings, gStr, q, rStr, l)
}

func TransFloat64ToInt64(K []float64) []int64 {
	result := make([]int64, len(K))
	for i := 0; i < len(K); i++ {
		result[i] = int64(K[i] * 1e5)
	}
	return result
}

func TotalBytes(slice []string) int {
	total := 0
	for _, s := range slice {
		total += len(s) // 每个字符串的字节数
	}
	return total
}
