package main

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"github.com/cbergoon/merkletree"
	"gnarktest/model"
	"gnarktest/pkg"
	"io"
	"log"
	"os"
	"path/filepath"
	"testing"
)

func GenerateRandomHash() (string, error) {
	randomBytes := make([]byte, 32)
	_, err := io.ReadFull(rand.Reader, randomBytes)
	if err != nil {
		return "", err
	}
	hash := sha256.Sum256(randomBytes)
	return hex.EncodeToString(hash[:]), nil
}
func areEqual(a, b [][]byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !bytes.Equal(a[i], b[i]) {
			return false
		}
	}
	return true
}
func TestMerkle(t *testing.T) {
	var list []merkletree.Content
	contents := []string{"Hello", "Hi", "Hey", "Hola"}
	list = append(list, model.MerkleContent{X: contents[0]})
	list = append(list, model.MerkleContent{X: contents[1]})
	list = append(list, model.MerkleContent{X: contents[2]})

	tree, err := merkletree.NewTree(list)
	if err != nil {
		log.Fatal(err)
	}

	root := tree.MerkleRoot()
	log.Println("root", root)

	proof, index, _ := tree.GetMerklePath(model.MerkleContent{X: contents[1]})
	fmt.Println(pkg.VerifyMerkleProof(model.MerkleContent{contents[1]}, proof, index, root, sha256.New))
}

func TestBench(t *testing.T) {
	data := make([]string, 345)
	list := make([]merkletree.Content, 345)
	for i := 0; i < 345; i++ {
		hash, _ := GenerateRandomHash()
		data[i] = hash
		list[i] = model.MerkleContent{X: hash}
	}
	fmt.Println(list)
	tree, err := merkletree.NewTree(list)
	if err != nil {
		log.Fatal(err)
	}
	root := tree.MerkleRoot()
	log.Println("root", root)
	rootStr := pkg.SerializeRoot(root)
	log.Println("rootStr", rootStr)

	proof, index, _ := tree.GetMerklePath(list[213])
	proofStrs := pkg.SerializeProof(proof)
	fmt.Println("proofStrs", proofStrs)

	fmt.Println(pkg.VerifyMerkleProofByString(data[213], proofStrs, index, rootStr, sha256.New))
	fmt.Println(pkg.VerifyMerkleProofByString(data[214], proofStrs, index, rootStr, sha256.New))
}

func TestReadJSONAndBuild(t *testing.T) {
	err := pkg.InitMerkleTrees()
	if err != nil {
		log.Fatal(err)
	}
	list, err := pkg.ReadDataSetFromFile("../law.json")
	if err != nil {
		log.Fatal(err)
	}
	path, index, root, genCost, _ := pkg.GenMerkleProof("law", list[10])
	fmt.Println("genCost", genCost)
	fmt.Println(pkg.VerifyMerkleProofByString(list[10], path, index, root, sha256.New))
	fmt.Println(pkg.VerifyMerkleProofByString(list[11], path, index, root, sha256.New))
}

func TestInitMerkleTrees(t *testing.T) {
	if err := pkg.InitMerkleTrees(); err != nil {
		log.Fatal(err)
	}
	l, _ := InitDataList()
	if err := pkg.InitMerkleTrees(); err != nil {
		log.Fatal(err)
	}
	merkleProof, index, root, cost2, err := pkg.GenMerkleProof("commonsense", l[0][0])
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("merkleProof", merkleProof)
	fmt.Println("index", index)
	fmt.Println("root", root)
	fmt.Println("cost2", cost2)
}
func InitDataList() ([][]string, error) {
	names, err := pkg.GetFileNames()
	if err != nil {
		return nil, err
	}
	res := make([][]string, 0)
	for _, name := range names {
		list := make([]string, 0)
		// 1. 打开JSONL文件
		file, err := os.Open(filepath.Join(filepath.Join("asset", "data"), name+".jsonl"))
		if err != nil {
			panic(err)
		}
		defer file.Close()

		// 2. 创建扫描器逐行读取
		scanner := bufio.NewScanner(file)
		lineNum := 0

		for scanner.Scan() {
			lineNum++
			line := scanner.Bytes() // 获取当前行的字节切片

			// 3. 解析JSON到结构体
			var p model.Obj
			if err := json.Unmarshal(line, &p); err != nil {
				fmt.Printf("解析第 %d 行失败: %v\n", lineNum, err)
				continue
			}
			list = append(list, p.PageContent)
		}
		res = append(res, list)
		fmt.Println(len(list))
	}
	return res, nil
}
