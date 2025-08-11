package model

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

type GenPoRReq struct {
	ClientId string    `json:"client_id"`
	K        []float64 `json:"K"`
	Q        []float64 `json:"Q"`
	Data     string    `json:"data"`
}

type VerifyPoRReq struct {
	ProofId int64
}

type GenPoGReq struct {
	Q    []float64 `json:"Q"`
	K    []float64 `json:"K"`
	Data string    `json:"data"`
}

type VerifyPoGReq struct {
	PoRId int64 `json:"por_id"`
	PoGId int64 `json:"pog_id"`
}

type PoR interface {
	Save2File() error
	LoadFromFile(int64) error
	GetBaseProof() BaseProof
}

type BaseProof struct {
	ID          int64       `json:"id"`
	MerkleProof MerkleProof `json:"merkle_proof"`
	Data        string      `json:"data"`
	Q           []float64   `json:"q"`
	L           int64       `json:"l"`
}

type CommitAndMerkleProof struct {
	BaseProof
	Commitment Commitment `json:"commitment"`
}

func (c *CommitAndMerkleProof) Save2File() error {
	file, err := os.Create(filepath.Join("./asset/proofs", fmt.Sprintf("%dpedersen.json", c.ID)))
	if err != nil {
		return err
	}
	defer file.Close()
	encoder := json.NewEncoder(file)
	if err = encoder.Encode(c); err != nil {
		return err
	}
	return nil
}

func (c *CommitAndMerkleProof) LoadFromFile(id int64) error {
	file, err := os.Open(filepath.Join("./asset/proofs", fmt.Sprintf("%dpedersen.json", id)))
	if err != nil {
		return err
	}
	defer file.Close()
	decoder := json.NewDecoder(file)
	if err = decoder.Decode(c); err != nil {
		return err
	}
	return nil
}

func (c *CommitAndMerkleProof) GetBaseProof() BaseProof {
	return c.BaseProof
}

type GrothAndMerkleProof struct {
	BaseProof
	GrothProof string `json:"groth_proof"`
}

func (g *GrothAndMerkleProof) Save2File() error {
	file, err := os.Create(filepath.Join("./asset/proofs", fmt.Sprintf("%dgroth.json", g.ID)))
	if err != nil {
		return err
	}
	defer file.Close()
	encoder := json.NewEncoder(file)
	if err = encoder.Encode(g); err != nil {
		return err
	}
	return nil
}

func (g *GrothAndMerkleProof) LoadFromFile(id int64) error {
	file, err := os.Open(filepath.Join("./asset/proofs", fmt.Sprintf("%dgroth.json", id)))
	if err != nil {
		return err
	}
	defer file.Close()
	decoder := json.NewDecoder(file)
	if err = decoder.Decode(g); err != nil {
		return err
	}
	return nil
}

func (g *GrothAndMerkleProof) GetBaseProof() BaseProof {
	return g.BaseProof
}

type Commitment struct {
	Commits []string `json:"commits"`
	R       string   `json:"r"`
}

type MerkleProof struct {
	Proof []string `json:"proof"`
	Index []int64  `json:"index"`
	Root  string   `json:"root"`
}

type PoG struct {
	ID       int64  `json:"id"`
	Distance int64  `json:"distance"`
	DataHash string `json:"data_hash"`
}

func (p *PoG) Save2File() error {
	file, err := os.Create(filepath.Join("./asset/proofs/PoG", fmt.Sprintf("%d.json", p.ID)))
	if err != nil {
		return err
	}
	defer file.Close()
	encoder := json.NewEncoder(file)
	if err = encoder.Encode(p); err != nil {
		return err
	}
	return nil
}
func (p *PoG) ReadFromFile(id int64) error {
	file, err := os.Open(filepath.Join("./asset/proofs/PoG", fmt.Sprintf("%d.json", id)))
	if err != nil {
		return err
	}
	defer file.Close()
	decoder := json.NewDecoder(file)
	if err = decoder.Decode(p); err != nil {
		return err
	}
	return nil
}
