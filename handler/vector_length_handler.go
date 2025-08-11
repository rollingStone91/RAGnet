package handler

import (
	"github.com/gin-gonic/gin"
	"gnarktest/config"
	"gnarktest/pkg"
	"net/http"
	"strconv"
)

// 获取当前向量长度
func GetCurrentVectorLengthHandler() func(c *gin.Context) {
	return func(c *gin.Context) {
		currentLength := config.GlobalVectorManager.GetCurrentLength()
		supportedLengths := config.GlobalVectorManager.GetSupportedLengths()

		c.JSON(http.StatusOK, gin.H{
			"code": 200,
			"data": gin.H{
				"current_length":    currentLength,
				"supported_lengths": supportedLengths,
			},
		})
	}
}

// 设置当前向量长度
func SetVectorLengthHandler() func(c *gin.Context) {
	return func(c *gin.Context) {
		lengthStr := c.Query("length")
		if lengthStr == "" {
			c.JSON(http.StatusBadRequest, gin.H{
				"code": 400,
				"msg":  "length parameter is required",
			})
			return
		}

		length, err := strconv.Atoi(lengthStr)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"code": 400,
				"msg":  "invalid length parameter",
			})
			return
		}

		// 检查是否支持该长度
		if !config.GlobalVectorManager.IsSupported(length) {
			c.JSON(http.StatusBadRequest, gin.H{
				"code":              400,
				"msg":               "unsupported vector length",
				"supported_lengths": config.GlobalVectorManager.GetSupportedLengths(),
			})
			return
		}

		// 尝试初始化该长度的电路和密钥
		if err := pkg.GetGroth16Manager().InitForLength(length); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"code":  500,
				"msg":   "failed to initialize circuit and keys for the specified length",
				"error": err.Error(),
			})
			return
		}

		// 设置当前向量长度
		if config.GlobalVectorManager.SetVectorLength(length) {
			c.JSON(http.StatusOK, gin.H{
				"code": 200,
				"msg":  "vector length set successfully",
				"data": gin.H{
					"current_length": length,
				},
			})
		} else {
			c.JSON(http.StatusInternalServerError, gin.H{
				"code": 500,
				"msg":  "failed to set vector length",
			})
		}
	}
}

// 生成指定长度的电路和密钥
func GenerateCircuitHandler() func(c *gin.Context) {
	return func(c *gin.Context) {
		lengthStr := c.Query("length")
		if lengthStr == "" {
			c.JSON(http.StatusBadRequest, gin.H{
				"code": 400,
				"msg":  "length parameter is required",
			})
			return
		}

		length, err := strconv.Atoi(lengthStr)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"code": 400,
				"msg":  "invalid length parameter",
			})
			return
		}

		// 检查是否支持该长度
		if !config.GlobalVectorManager.IsSupported(length) {
			c.JSON(http.StatusBadRequest, gin.H{
				"code":              400,
				"msg":               "unsupported vector length",
				"supported_lengths": config.GlobalVectorManager.GetSupportedLengths(),
			})
			return
		}

		// 生成电路和密钥
		if err := pkg.GenerateCircuitAndKeys(length); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"code":  500,
				"msg":   "failed to generate circuit and keys",
				"error": err.Error(),
			})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"code": 200,
			"msg":  "circuit and keys generated successfully",
			"data": gin.H{
				"length": length,
			},
		})
	}
}

// 生成所有支持长度的电路和密钥
func GenerateAllCircuitsHandler() func(c *gin.Context) {
	return func(c *gin.Context) {
		if err := pkg.GenerateAllCircuitsAndKeys(); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"code":  500,
				"msg":   "failed to generate all circuits and keys",
				"error": err.Error(),
			})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"code": 200,
			"msg":  "all circuits and keys generated successfully",
			"data": gin.H{
				"supported_lengths": config.GlobalVectorManager.GetSupportedLengths(),
			},
		})
	}
}

// 测试指定长度的证明生成和验证
func TestVectorLengthHandler() func(c *gin.Context) {
	return func(c *gin.Context) {
		lengthStr := c.Query("length")
		if lengthStr == "" {
			c.JSON(http.StatusBadRequest, gin.H{
				"code": 400,
				"msg":  "length parameter is required",
			})
			return
		}

		length, err := strconv.Atoi(lengthStr)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"code": 400,
				"msg":  "invalid length parameter",
			})
			return
		}

		// 检查是否支持该长度
		if !config.GlobalVectorManager.IsSupported(length) {
			c.JSON(http.StatusBadRequest, gin.H{
				"code":              400,
				"msg":               "unsupported vector length",
				"supported_lengths": config.GlobalVectorManager.GetSupportedLengths(),
			})
			return
		}

		// 生成随机向量
		q, k, l := pkg.GenRandVector(length)

		// 生成证明
		proof, genCost, err := pkg.GenGrothProofWithLength(q, k, length)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"code":  500,
				"msg":   "failed to generate proof",
				"error": err.Error(),
			})
			return
		}

		// 验证证明
		valid, verifyCost, err := pkg.VerifyGrothProofWithLength(proof, q, l, length)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"code":  500,
				"msg":   "failed to verify proof",
				"error": err.Error(),
			})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"code": 200,
			"msg":  "test completed successfully",
			"data": gin.H{
				"length":      length,
				"proof_valid": valid,
				"generation_cost": gin.H{
					"time_ms":     genCost.Time.Milliseconds(),
					"space_bytes": genCost.Space,
				},
				"verification_cost": gin.H{
					"time_ms":     verifyCost.Time.Milliseconds(),
					"space_bytes": verifyCost.Space,
				},
			},
		})
	}
}
