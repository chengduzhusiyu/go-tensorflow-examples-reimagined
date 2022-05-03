
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
	"image/color"
	"image/png"
	"io/ioutil"
	"log"
	"os"
	"reflect"
	"unsafe"

	"github.com/disintegration/imaging"
	imagetypes "github.com/rai-project/image/types"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"golang.org/x/image/colornames"
	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

// DRAWING UTILITY FUNCTIONS

// HLine draws a horizontal line
func HLine(img *image.RGBA, x1, y, x2 int, col color.Color) {
	for ; x1 <= x2; x1++ {
		img.Set(x1, y, col)
	}
}

// VLine draws a veritcal line
func VLine(img *image.RGBA, x, y1, y2 int, col color.Color) {
	for ; y1 <= y2; y1++ {
		img.Set(x, y1, col)
	}
}

// Rect draws a rectangle utilizing HLine() and VLine()
func Rect(img *image.RGBA, x1, y1, x2, y2, width int, col color.Color) {
	for i := 0; i < width; i++ {
		HLine(img, x1, y1+i, x2, col)
		HLine(img, x1, y2+i, x2, col)
		VLine(img, x1+i, y1, y2, col)
		VLine(img, x2+i, y1, y2, col)
	}
}

// Segment draws a rectangle utilizing HLine() and VLine()
func Segment(img *image.RGBA, mask [][]float32, col color.Color, x1, y1, x2, y2 float32) *image.RGBA {
	height := len(mask)
	width := len(mask[0])
	seg := image.NewRGBA(image.Rect(0, 0, width, height))

	for ii := 0; ii < height; ii++ {
		for jj := 0; jj < width; jj++ {
			if mask[ii][jj] > 0.2 {
				seg.Set(jj, ii, col)
			}
		}
	}

	segScaled := imaging.Resize(seg, int(x2)-int(x1), int(y2)-int(y1), imaging.NearestNeighbor)

	out, _ := os.Create("/tmp/test.png")
	defer out.Close()
	err := png.Encode(out, segScaled)
	if err != nil {
		log.Println(err)
	}

	overlay := imaging.Overlay(img, segScaled, image.Pt(int(x1), int(y1)), 0.5)
	rgba := &image.RGBA{
		Pix:    overlay.Pix,
		Stride: overlay.Stride,
		Rect:   overlay.Rect,
	}

	return rgba
}

func ToPng(filePath string, imgByte []byte, bounds image.Rectangle) {
	img := imagetypes.NewRGBImage(bounds)
	copy(img.Pix, imgByte)

	out, _ := os.Create(filePath)
	defer out.Close()

	err := png.Encode(out, img.ToRGBAImage())
	if err != nil {
		log.Println(err)
	}
}

// LABEL UTILITY FUNCTIONS

func LoadLabels(labelsFile string) []string {
	var labels []string
	file, err := os.Open(labelsFile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		log.Printf("ERROR: failed to read %s: %v", labelsFile, err)
	}

	return labels
}

func GetLabel(idx int, probabilities []float32, classes []float32, labels []string) string {
	index := int(classes[idx])
	label := fmt.Sprintf("%s (%2.0f%%)", labels[index], probabilities[idx]*100.0)

	return label
}

func AddLabel(img *image.RGBA, x, y, class int, label string) {
	col := colornames.Map[colornames.Names[class]]
	point := fixed.Point26_6{fixed.Int26_6(x * 64), fixed.Int26_6(y * 64)}

	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(colornames.Black),
		Face: basicfont.Face7x13,
		Dot:  point,
	}

	Rect(img, x, y-13, (x + len(label)*7), y-6, 7, col)

	d.DrawString(label)
}

// TENSOR UTILITY FUNCTIONS
