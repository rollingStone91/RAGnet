package model

import (
	"github.com/consensys/gnark/frontend"
)

type InnerProductCircuit struct {
	A       []frontend.Variable `gnark:",public"`
	B       []frontend.Variable `gnark:",secret"`
	Desired frontend.Variable   `gnark:",public"`
}

func (circuit *InnerProductCircuit) Define(api frontend.API) error {
	sum := api.Add(0, 0)
	for i := 0; i < len(circuit.A); i++ {
		product := api.Mul(circuit.A[i], circuit.B[i])
		sum = api.Add(sum, product)
	}
	api.AssertIsEqual(sum, circuit.Desired)
	return nil
}
