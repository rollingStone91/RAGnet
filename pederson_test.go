package main

import (
	"fmt"
	"gnarktest/pkg"
	"testing"
)

var (
	k = []int64{97, 72, 36, 63, 79, 19, 36, 17, 66, 20, 85, 42, 85, 27, 69, 9, 52, 43, 29, 88, 28, 54, 49, 9, 21, 84,
		87, 10, 5, 4, 54, 14, 75, 70, 68, 89, 60, 5, 10, 77, 94, 85, 29, 68, 47, 47, 69, 98, 27, 15, 84, 52, 39, 91, 39, 40, 53,
		46, 21, 67, 69, 73, 2, 74, 36, 38, 94, 50, 22, 35, 47, 2, 67, 90, 73, 60, 90, 94, 44, 83, 59, 75, 48, 24, 6, 73, 97, 73,
		68, 28, 13, 97, 71, 31, 65, 17, 48, 10, 80, 31, 54, 23, 16, 10, 25, 43, 27, 54, 26, 81, 71, 19, 63, 48, 65, 41, 8, 58, 22,
		3, 40, 77, 69, 46, 85, 34, 33, 52, 41, 77, 40, 78, 92, 57, 37, 14, 26, 87, 90, 44, 78, 41, 92, 30, 44, 0, 27, 81, 90, 94,
		64, 18, 8, 52, 54, 39, 19, 57, 56, 64, 94, 96, 6, 98, 96, 26, 77, 57, 32, 74, 38, 39, 30, 85, 82, 65, 53, 43, 13, 95, 31,
		87, 12, 52, 63, 92, 75, 78, 75, 95, 87, 64, 78, 63, 77, 78, 50, 24, 31, 92, 5, 77, 66, 80, 6, 36, 41, 8, 77, 53, 40, 70, 35,
		72, 58, 62, 48, 52, 65, 83, 63, 37, 71, 33, 9, 15, 24, 75, 67, 45, 75, 9, 50, 87, 76, 41, 36, 94, 4, 84, 64, 6, 2, 57, 15, 20, 4, 64, 7, 10, 75, 1, 58, 6, 70, 96,
	}
	q = []int64{40, 60, 8, 95, 13, 42, 38, 84, 30, 38, 36, 37, 71, 73, 30, 69, 58, 93, 98, 6, 89, 31, 16, 5, 30, 90, 9, 99, 54,
		41, 6, 59, 43, 69, 18, 37, 56, 96, 87, 40, 65, 94, 56, 5, 40, 39, 46, 67, 73, 19, 36, 39, 1, 20, 26, 91, 39, 15, 13, 47, 17,
		0, 19, 65, 36, 56, 44, 32, 73, 9, 44, 85, 23, 91, 42, 88, 67, 64, 39, 25, 53, 76, 80, 88, 98, 69, 30, 75, 5, 24, 29, 16, 10,
		45, 16, 63, 26, 96, 36, 92, 18, 61, 41, 69, 17, 28, 66, 74, 93, 30, 82, 44, 89, 19, 84, 48, 82, 53, 94, 72, 62, 61, 1, 58, 22,
		34, 63, 76, 10, 14, 94, 5, 58, 52, 63, 23, 27, 95, 80, 21, 37, 35, 84, 51, 77, 37, 43, 17, 0, 63, 96, 77, 77, 32, 19, 20, 14,
		68, 90, 45, 50, 44, 31, 55, 30, 51, 32, 40, 10, 67, 70, 19, 25, 95, 93, 47, 83, 57, 58, 61, 72, 71, 96, 71, 26, 35, 20, 85,
		8, 26, 62, 84, 84, 97, 96, 43, 89, 62, 6, 66, 7, 91, 28, 2, 14, 3, 59, 40, 2, 52, 26, 74, 9, 76, 36, 3, 32, 97, 18, 12, 20, 48,
		58, 72, 5, 81, 30, 72, 95, 62, 74, 15, 55, 20, 61, 23, 78, 65, 99, 93, 40, 54, 59, 24, 16, 7, 82, 40, 35, 47, 50, 59, 41, 97, 18, 48,
	}
	l = int64(634017)
)

func TestName(t *testing.T) {
	K, Q, L := pkg.GenRandVector(12)
	com, r, _, _ := pkg.GenCommitment(K)
	res, _, err := pkg.VerifyCommitment(com, Q, L, r)
	if err != nil || res == false {
		fmt.Println(err)
	}
}

func Test(t *testing.T) {
	g := pkg.ParamsGen()
	r := pkg.RandomGen()

	coms, genCost, _ := pkg.Commit(g, k, r)
	fmt.Println("genCost:", genCost)
	res, verifyCost := pkg.Check(coms, g, r, q, l)
	fmt.Println("verifyCost:", verifyCost)
	fmt.Println("res:", res)
	q[0] = 1
	res, verifyCost1 := pkg.Check(coms, g, r, q, l)
	fmt.Println("verifyCost1:", verifyCost1)
	fmt.Println("res1:", res)
}

func TestRealDataInPedersen(t *testing.T) {
	q, k, _ := ReadKQFromDemo()
	Q := pkg.TransFloat64ToInt64(q)
	K := pkg.TransFloat64ToInt64(k)
	L := pkg.CalculateInnerProduct(Q, K)
	g, _ := pkg.GetGenerateElement()
	r := pkg.RandomGenToNumberString()
	cs, genCost, _ := pkg.CommitFromString(g, K, r)
	fmt.Println("genCost:", genCost)
	res, verifyCost, _ := pkg.CheckByString(cs, g, Q, r, L)
	fmt.Println("verifyCost:", verifyCost)
	fmt.Println("res:", res)
	res, verifyCost1, _ := pkg.CheckByString(cs, g, Q, r, L)
	fmt.Println("verifyCost1:", verifyCost1)
	fmt.Println("res1:", res)
}
