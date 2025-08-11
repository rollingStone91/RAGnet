package main

import (
	"encoding/json"
	"fmt"
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"gnarktest/model"
	"gnarktest/pkg"
	"log"
	"os"
	"testing"
)

/*func TestOrigin(t *testing.T) {
	a := make([]frontend.Variable, 256)
	b := make([]frontend.Variable, 256)

	// 初始化向量（这里用0填充，实际使用时会被替换）
	for i := 0; i < 256; i++ {
		a[i] = 0
		b[i] = 0
	}

	circuit := model.InnerProductCircuit{
		A:       a,
		B:       b,
		Desired: 0,
	}
	ccs, _ := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &circuit)

	pk, vk, _ := groth16.Setup(ccs)

	a, b, res := pkg.GenRandVector(config.VectorLength)
	fmt.Println("len(a)", len(a))
	assignment := model.InnerProductCircuit{
		A:       pkg.ConvertToElement(a),
		B:       pkg.ConvertToElement(b),
		Desired: res,
	}

	//构造witness
	witness, _ := frontend.NewWitness(&assignment, ecc.BN254.ScalarField())

	//生成证明
	start := time.Now()
	proof, err := groth16.Prove(ccs, pk, witness)

	elapsed := time.Since(start)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Prover time: %s\n", elapsed)

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer func() {
			if r := recover(); r != nil {
				fmt.Printf("goroutine panic: %v\n", r)
			}
		}()
		var buf bytes.Buffer
		_, err := proof.WriteTo(&buf)
		if err != nil {
			panic(err)
		}
		log.Printf("证明大小为 %d 字节\n", buf.Len())
	}()
	//验证者自己通过公共输入生成publicWitness
	publicInput := model.InnerProductCircuit{
		A:       pkg.ConvertToElement(a),
		Desired: res,
	}
	pWitness, _ := frontend.NewWitness(&publicInput, ecc.BN254.ScalarField(), frontend.PublicOnly())
	start = time.Now()
	err = groth16.Verify(proof, vk, pWitness)
	verifyElapsed := time.Since(start)
	fmt.Printf("Verifier time: %s\n", verifyElapsed)
	if err != nil {
		log.Println("证明失败", err)
	} else {
		log.Println("证明成功")
	}
	wg.Wait()
}*/

func SaveData(as *model.InnerProductCircuit) error {
	data, err := json.Marshal(as)
	if err != nil {
		return err
	}
	err = os.WriteFile("asset/circuit.json", data, 0644)
	if err != nil {
		return err
	}
	return nil
}

// 会报错不知道为什么，读出来的对象看起来是一样的，但是用在验证中会报错
func ReadPublicInput() (model.InnerProductCircuit, error) {
	data, err := os.ReadFile("asset/circuit.json")
	if err != nil {
		return model.InnerProductCircuit{}, err
	}
	var as model.InnerProductCircuit
	err = json.Unmarshal(data, &as)
	if err != nil {
		return model.InnerProductCircuit{}, err
	}
	res := model.InnerProductCircuit{
		A:       as.A,
		Desired: as.Desired,
	}
	return res, nil
}

func TestStartFromFile(t *testing.T) {
	ccs, err := pkg.ReadCcs("./asset/groth/cs")
	if err != nil {
		panic(err)
	}
	q, k, _ := ReadKQFromDemo()
	Q := pkg.TransFloat64ToInt64(q)
	K := pkg.TransFloat64ToInt64(k)
	L := pkg.CalculateInnerProduct(Q, K)
	pk, err := pkg.ReadPk("./asset/groth/proving.key")
	if err != nil {
		panic(err)
	}
	assignment := model.InnerProductCircuit{
		A:       pkg.ConvertToElement(Q),
		B:       pkg.ConvertToElement(K),
		Desired: L,
	}
	witness, _ := frontend.NewWitness(&assignment, ecc.BN254.ScalarField())

	proof, err := groth16.Prove(ccs, pk, witness)
	if err != nil {
		panic(err)
	}
	publicInput := model.InnerProductCircuit{
		A:       pkg.ConvertToElement(Q),
		Desired: L,
	}
	publicWitness, err := frontend.NewWitness(&publicInput, ecc.BN254.ScalarField(), frontend.PublicOnly())
	if err != nil {
		panic(err)
	}
	vk, _ := pkg.ReadVk("./asset/groth/verifying.key")
	err = groth16.Verify(proof, vk, publicWitness)
	if err != nil {
		fmt.Println("验证失败")
	} else {
		fmt.Println("验证成功")
	}
}

// 验证方用public input构造public witness，结合proof和vk进行验证
func TestReadAndVerify(t *testing.T) {
	publicInput := model.InnerProductCircuit{
		A: pkg.ConvertToElement([]int64{81, 96, 95, 49, 30, 16, 60, 27, 90, 55, 84, 93, 1, 86, 75, 82, 88, 38, 5, 92, 38, 19,
			79, 64, 40, 80, 66, 5, 20, 23, 13, 16, 20, 11, 14, 20, 78, 31, 95, 5, 17, 53, 9, 45, 78, 10, 18, 50, 88, 77, 10, 89, 43, 93, 55,
			67, 47, 12, 91, 26, 98, 40, 20, 69, 7, 95, 94, 17, 71, 18, 7, 37, 23, 99, 64, 14, 8, 49, 81, 30, 40, 87, 40, 40, 76, 72, 21, 53,
			83, 46, 98, 74, 91, 3, 97, 65, 33, 38, 32, 20, 87, 2, 62, 75, 47, 16, 0, 52, 91, 97, 77, 79, 85, 78, 85, 8, 10, 26, 45, 91, 52, 76,
			91, 4, 5, 96, 6, 64, 12, 2, 55, 46, 25, 52, 94, 54, 45, 54, 59, 91, 17, 41, 60, 7, 73, 62, 26, 57, 11, 11, 51, 93, 10, 55, 23, 72,
			81, 63, 20, 66, 15, 24, 93, 22, 62, 89, 70, 16, 91, 69, 93, 86, 73, 70, 30, 42, 61, 10, 63, 34, 29, 68, 66, 93, 76, 15, 77, 71,
			33, 70, 10, 80, 71, 30, 12, 9, 4, 96, 92, 55, 47, 71, 79, 3, 33, 80, 71, 94, 32, 44, 24, 24, 48, 76, 80, 28, 74, 11, 27, 61, 30, 24,
			27, 56, 31, 85, 84, 51, 31, 94, 38, 22, 63, 78, 86, 5, 51, 60, 4, 34, 91, 48, 58, 18, 5, 60, 23, 44, 43, 84, 60, 44, 61, 73, 20, 95,
		}),
		Desired: 683362,
	}
	fmt.Println("publicInput:", publicInput)
	publicWitness, _ := frontend.NewWitness(&publicInput, ecc.BN254.ScalarField(), frontend.PublicOnly())

	proof, err := pkg.ReadProof("./asset/proof")
	if err != nil {
		t.Fatal(err)
	}
	vk, err := pkg.ReadVk("./asset/keys/verifying.key")
	if err != nil {
		t.Fatal(err)
	}

	if err = groth16.Verify(proof, vk, publicWitness); err != nil {
		log.Fatal(err)
	} else {
		fmt.Println("验证成功")
	}
}

func TestHandler(t *testing.T) {
	pkg.InitGroth()
	q, k, l := pkg.GenRandVector(256)
	fmt.Println("q:", q)
	fmt.Println("k:", k)
	proof, genCost, _ := pkg.GenGrothProof(q, k)
	fmt.Println("proof:", proof)
	fmt.Println("genCost:", &genCost)
	res, verifyCost, _ := pkg.VerifyGrothProof(proof, q, l)
	if res {
		fmt.Println("验证成功")
	} else {
		fmt.Println("验证失败")
	}
	fmt.Println("verifyCost:", &verifyCost)
	res, verifyCost1, _ := pkg.VerifyGrothProof(proof, q, l+1)
	if res {
		fmt.Println("验证成功")
	} else {
		fmt.Println("验证失败")
	}
	fmt.Println("verifyCost1:", &verifyCost1)
}

func TestRealData(t *testing.T) {
	pkg.InitGroth()
	q, k, l := ReadKQFromDemo()
	Q := pkg.TransFloat64ToInt64(q)
	K := pkg.TransFloat64ToInt64(k)
	L := pkg.CalculateInnerProduct(Q, K)
	L1 := int64(l * 1e10)
	//todo:先计算内积再转化为int64与先转化再计算内积产生的结果不一样
	fmt.Println("L", L)
	fmt.Println("L1:", L1)
	proof, genCost, _ := pkg.GenGrothProof(Q, K)
	fmt.Printf("genCost:%+v\n", genCost)
	res, verifyCost, _ := pkg.VerifyGrothProof(proof, Q, L)
	fmt.Printf("verifyCost:%+v\n", verifyCost)
	if res {
		fmt.Println("验证成功")
	} else {
		fmt.Println("验证失败")
	}
	res, verifyCost1, _ := pkg.VerifyGrothProof(proof, Q, L+1)
	fmt.Printf("verifyCost1:%+v\n", verifyCost1)

	if res {
		fmt.Println("验证成功")
	} else {
		fmt.Println("验证失败")
	}
}

func TestInitCircuit(t *testing.T) {
	var circuit model.InnerProductCircuit
	ccs, _ := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &circuit)

	pk, vk, _ := groth16.Setup(ccs)
	err := pkg.SaveGrothKeys(pk, vk)
	if err != nil {
		t.Fatal(err)
	}
	err = pkg.SaveCcs(ccs)
	if err != nil {
		t.Fatal(err)
	}
	pkg.InitGroth()
}
