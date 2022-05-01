
package utils

// #include <stdlib.h>
// #cgo LDFLAGS: -ltensorflow
// #cgo CFLAGS: -I${SRCDIR}/../../tensorflow/tensorflow
// #include "tensorflow/c/c_api.h"
import "C"

import (
	"bufio"
	"bytes"
	"fmt"
	"image"