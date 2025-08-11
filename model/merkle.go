package model

import (
	"crypto/sha256"
	"github.com/cbergoon/merkletree"
)

type Obj struct {
	PageContent string `json:"page_content"`
}

type MerkleContent struct {
	X string
}

func (t MerkleContent) CalculateHash() ([]byte, error) {
	h := sha256.New()
	if _, err := h.Write([]byte(t.X)); err != nil {
		return nil, err
	}

	return h.Sum(nil), nil
}

func (t MerkleContent) Equals(other merkletree.Content) (bool, error) {
	otherTC, ok := other.(MerkleContent)
	if !ok {
		return false, nil
	}
	return t.X == otherTC.X, nil
}
