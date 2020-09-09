
package main

// #cgo LDFLAGS: -ltensorflow
// #cgo CFLAGS: -I${SRCDIR}/../../../tensorflow/tensorflow/
// #include <stdlib.h>
// #include "tensorflow/c/c_api.h"
import "C"

import (
	// "path/filepath"

	"unsafe"
	"github.com/k0kubun/pp"
	// utils "github.com/rai-project/tensorflow-go-examples"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

var (
	dir = "/home/abduld/mlperf/inference/v0.5/translation/gnmt/tensorflow/savedmodel"
)

// translate.ckpt.data-00000-of-00001  translate.ckpt.index  translate.ckpt.meta

func main() {


	pth := C.CString("_beam_search_ops.so")
	C.free(unsafe.Pointer(pth))
	stat := C.TF_NewStatus()
	C.TF_LoadLibrary(pth, stat);