package config

const (
	BasePath     = "asset"
	MerklePath   = "data"
	GrothPath    = "groth"
	GrothCcsPath = "cs"
	GrothPkPath  = "proving.key"
	GrothVkPath  = "verifying.key"
	PedersenPath = "pedersen/g"
	VectorLength = 256
)

var VectorLengthList = []int{128, 256, 384, 512, 640, 768, 896, 1024, 2048}

// 动态向量长度管理器
type VectorLengthManager struct {
	CurrentLength    int
	SupportedLengths []int
}

var GlobalVectorManager = &VectorLengthManager{
	CurrentLength:    VectorLength,
	SupportedLengths: VectorLengthList,
}

// 设置当前向量长度
func (vm *VectorLengthManager) SetVectorLength(length int) bool {
	for _, supported := range vm.SupportedLengths {
		if supported == length {
			vm.CurrentLength = length
			return true
		}
	}
	return false
}

// 获取当前向量长度
func (vm *VectorLengthManager) GetCurrentLength() int {
	return vm.CurrentLength
}

// 获取支持的向量长度列表
func (vm *VectorLengthManager) GetSupportedLengths() []int {
	return vm.SupportedLengths
}

// 检查是否支持指定的向量长度
func (vm *VectorLengthManager) IsSupported(length int) bool {
	for _, supported := range vm.SupportedLengths {
		if supported == length {
			return true
		}
	}
	return false
}
