package model

import "time"

type Cost struct {
	Time  time.Duration // 时间开销
	Space int64         // 空间开销（单位：字节）
}
