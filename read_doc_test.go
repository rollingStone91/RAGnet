package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"
)

func CaculateInnerProduct(v1, v2 []float64) float64 {
	res := float64(0)
	for i := 0; i < len(v1); i++ {
		res += v1[i] * v2[i]
	}
	return res
}

func ReadKQFromDemo() ([]float64, []float64, float64) {
	file, err := os.Open("../data.txt")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	var vectors [][]float64
	var vector []float64
	scanner := bufio.NewScanner(file)
	start1 := false
	start2 := false
	i := 0
	maxNum := float64(0)
	for scanner.Scan() {
		i++
		line := scanner.Text()
		if strings.HasPrefix(line, "[") {
			num, err := strconv.ParseFloat(line[1:len(line)-1], 64)
			if err != nil {
				continue
			}
			vector = append(vector, num)
			start1 = true
			continue
		} else if strings.HasPrefix(line, "[[") || strings.HasPrefix(line, " [") {
			start2 = true
			num, err := strconv.ParseFloat(line[2:len(line)-1], 64)
			if err != nil {
				panic(err)
			}
			vector = append(vector, num)
			continue
		}
		if start1 || start2 {
			if strings.HasSuffix(line, "]") || strings.HasSuffix(line, "],") {
				start1 = false
				start2 = false
				temp := make([]float64, len(vector))
				copy(temp, vector)
				vectors = append(vectors, temp)
				vector = nil
				continue
			}
			num := float64(0)
			if start1 {
				num, err = strconv.ParseFloat(line[1:len(line)-1], 64)
			} else {
				num, err = strconv.ParseFloat(line[2:len(line)-1], 64)
			}
			maxNum = math.Max(maxNum, math.Abs(num))
			if err != nil {
				fmt.Println(line)
				panic(err)
			}
			vector = append(vector, num)
		}
	}
	return vectors[1], vectors[2], CaculateInnerProduct(vectors[1], vectors[2])
}

func TestReadFromDoc(t *testing.T) {
	file, err := os.Open("../data.txt")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	var vectors [][]float64
	var vector []float64
	scanner := bufio.NewScanner(file)
	start1 := false
	start2 := false
	i := 0
	maxNum := float64(0)
	for scanner.Scan() {
		i++
		line := scanner.Text()
		if strings.HasPrefix(line, "[") {
			fmt.Println("line", i)
			num, err := strconv.ParseFloat(line[1:len(line)-1], 64)
			if err != nil {
				continue
			}
			vector = append(vector, num)
			start1 = true
			continue
		} else if strings.HasPrefix(line, "[[") || strings.HasPrefix(line, " [") {
			fmt.Println("line", i)
			start2 = true
			num, err := strconv.ParseFloat(line[2:len(line)-1], 64)
			if err != nil {
				panic(err)
			}
			vector = append(vector, num)
			continue
		}
		if start1 || start2 {
			if strings.HasSuffix(line, "]") || strings.HasSuffix(line, "],") {
				start1 = false
				start2 = false
				temp := make([]float64, len(vector))
				copy(temp, vector)
				vectors = append(vectors, temp)
				vector = nil
				continue
			}
			num := float64(0)
			if start1 {
				num, err = strconv.ParseFloat(line[1:len(line)-1], 64)
			} else {
				num, err = strconv.ParseFloat(line[2:len(line)-1], 64)
			}
			maxNum = math.Max(maxNum, math.Abs(num))
			if err != nil {
				fmt.Println(line)
				panic(err)
			}
			vector = append(vector, num)
		}
	}
	fmt.Println(CaculateInnerProduct(vectors[1], vectors[2]))
}

func TestLine(t *testing.T) {
	file, err := os.Open("../data.txt")
	if err != nil {
		panic(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasSuffix(line, "],") {
			fmt.Println(line)
		}
	}
}
