package pkg

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"github.com/cbergoon/merkletree"
	"gnarktest/config"
	"gnarktest/model"
	"gnarktest/utils"
	"hash"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"time"
)

var (
	treeMap    = make(map[string]*merkletree.MerkleTree)
	StringsMap = make(map[string][]string)
)

type Document struct {
	PageContent string `json:"page_content"`
}

func GetFileNames() ([]string, error) {
	var fileNames []string

	// 读取目录内容
	entries, err := os.ReadDir(filepath.Join(config.BasePath, config.MerklePath))
	if err != nil {
		return nil, err
	}

	// 遍历目录项
	for _, entry := range entries {
		if !entry.IsDir() { // 只处理文件，忽略子目录
			nameWithoutExt := strings.TrimSuffix(entry.Name(), ".json")
			fileNames = append(fileNames, nameWithoutExt)
		}
	}
	return fileNames, nil
}

// Handler调用，要确保client_id与文件名（不加后缀）相同
func InitMerkleTrees() error {
	names, err := GetFileNames()
	if err != nil {
		return err
	}
	for _, name := range names {
		data, err := ioutil.ReadFile(filepath.Join(filepath.Join(config.BasePath, config.MerklePath), name+".json"))
		if err != nil {
			panic("读取文件失败:")
		}
		// 解析JSON数据
		var documents map[string]Document
		err = json.Unmarshal(data, &documents)
		if err != nil {
			panic("解析JSON失败:")
		}
		// 创建字符串切片来存储所有的page_content
		var pageContents []string
		// 遍历所有的文档，提取page_content
		for _, doc := range documents {
			pageContents = append(pageContents, doc.PageContent)
		}

		tree, err := InitTreeFromList(pageContents)
		if err != nil {
			return err
		}
		treeMap[name] = tree
		StringsMap[name] = pageContents
	}
	return nil
}

func InitTreeFromList(strList []string) (*merkletree.MerkleTree, error) {
	list := make([]merkletree.Content, len(strList))
	for i := 0; i < len(strList); i++ {
		list[i] = model.MerkleContent{X: strList[i]}
	}
	tree, err := merkletree.NewTree(list)
	if err != nil {
		return nil, err
	}
	return tree, nil
}

// Handler调用
func GenMerkleProof(clientId string, data string) ([]string, []int64, string, *model.Cost, error) {
	tree := treeMap[clientId]
	cost := &model.Cost{}

	start := time.Now()
	proof, index, err := tree.GetMerklePath(model.MerkleContent{X: data})
	root := tree.MerkleRoot()
	elapsed := time.Since(start)
	cost.Time = elapsed
	if err != nil {
		return nil, nil, "", nil, err
	}
	cost.Space = utils.CountSpace(proof) + int64(len(index)*8) + int64(len(root)) + int64(len([]byte(data)))
	return SerializeProof(proof), index, SerializeRoot(root), cost, nil
}

// Handler调用
func VerifyMerkleProofByString(data string, proofStrings []string, index []int64, rootString string, hashFunc func() hash.Hash) (*model.Cost, bool) {
	content := model.MerkleContent{X: data}
	proof := DeSerializeProof(proofStrings)
	root := DeSerializeRoot(rootString)
	return VerifyMerkleProof(content, proof, index, root, hashFunc)
}

func SerializeProof(proof [][]byte) []string {
	res := make([]string, len(proof))
	for i, p := range proof {
		res[i] = base64.StdEncoding.EncodeToString(p)
	}
	return res
}

func DeSerializeProof(proofStrs []string) [][]byte {
	res := make([][]byte, len(proofStrs))
	for i, p := range proofStrs {
		res[i], _ = base64.StdEncoding.DecodeString(p)
	}
	return res
}

func SerializeRoot(root []byte) string {
	return base64.StdEncoding.EncodeToString(root)
}

func DeSerializeRoot(rootStr string) []byte {
	res, _ := base64.StdEncoding.DecodeString(rootStr)
	return res
}

func VerifyMerkleProof(content merkletree.Content, proof [][]byte, index []int64, root []byte, hashFunc func() hash.Hash) (*model.Cost, bool) {
	start := time.Now()
	contentHash, _ := content.CalculateHash()

	for i, p := range proof {
		h := hashFunc()
		if index[i]%2 == 1 {
			h.Write(contentHash)
			h.Write(p)
		} else {
			h.Write(p)
			h.Write(contentHash)
		}
		contentHash = h.Sum(nil)
	}
	res := bytes.Equal(contentHash, root)
	elapsed := time.Since(start)
	cost := new(model.Cost)
	cost.Time = elapsed
	return cost, res
}

func ReadDataSetFromFile(path string) ([]string, error) {
	// 读取文件内容
	fileContent, err := os.ReadFile(path)
	if err != nil {
		fmt.Println("读取文件失败:", err)
		return nil, err
	}

	// 定义一个变量来存储解析后的数据
	var data [][]string

	// 解析JSON内容
	err = json.Unmarshal(fileContent, &data)
	if err != nil {
		fmt.Println("解析JSON失败:", err)
		return nil, err
	}
	list := make([]string, len(data))
	for i, d := range data {
		for _, p := range d {
			list[i] += p
		}
	}
	return list, nil
}

/*// Handler调用，要确保client_id与文件名（不加后缀）相同
func InitMerkleTrees() error {
	names, err := GetFileNames()
	if err != nil {
		return err
	}
	for _, name := range names {
		list := make([]string, 0)
		// 1. 打开JSONL文件
		file, err := os.Open(filepath.Join(filepath.Join(config.BasePath, config.MerklePath), name+".jsonl"))
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
		tree, err := InitTreeFromList(list)
		if err != nil {
			return err
		}
		treeMap[name] = tree
		StringsMap[name] = list
	}
	return nil
}*/
