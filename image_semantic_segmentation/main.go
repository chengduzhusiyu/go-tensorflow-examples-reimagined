
package main

import (
	"flag"
	"image"
	"image/color"
	"image/jpeg"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"

	"github.com/disintegration/imaging"

	utils "github.com/rai-project/tensorflow-go-examples"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

var LabelNames []string = []string{"background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
	"car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
	"sheep", "sofa", "train", "tv"}

func createPascalLabelColorMap() [256][3]int32 {
	var colorMap [256][3]int32
	var ind [256]int32
	for ii := 0; ii < 256; ii++ {
		ind[ii] = int32(ii)
	}
	for shift := 7; shift >= 0; shift-- {
		for jj := 0; jj < 256; jj++ {
			for kk := 0; kk < 3; kk++ {
				colorMap[jj][kk] |= ((ind[jj] >> uint(kk)) & 1) << uint(shift)
			}
		}
		for jj := range ind {
			ind[jj] >>= 3
		}
	}
	return colorMap
}

func main() {
	// Parse flags
	modeldir := flag.String("dir", "", "Directory containing trained model files. Assumes model file is called frozen_inference_graph.pb")
	jpgfile := flag.String("jpg", "lane_control.jpg", "Path of a JPG image to use for input")
	outjpg := flag.String("out", "output.jpg", "Path of output JPG for displaying labels. Default is output.jpg")
	flag.Parse()
	if *modeldir == "" || *jpgfile == "" {