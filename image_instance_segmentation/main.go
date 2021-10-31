package main

import (
	"flag"
	"image"
	"image/draw"
	"image/jpeg"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"

	utils "github.com/rai-project/tensorflow-go-examples"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"golang.org/x/image/colornames"
)

func main() {
	// Parse flags
	modeldir := flag.String("dir", "", "Directory containing trained model files. Assumes model file is called frozen_inference_graph.pb")
	jpgfile := flag.String("jpg", "lane_control.jpg", "Path of a JPG image to use for input")
	outjpg := flag.String("out", "output.jpg", "Path of output JPG for displaying labels. Default is output.jpg")
	labelfile := flag.String("labels", "coco_labels.txt", "Path to file of COCO labels, one per line")
	flag.Parse()
	if *mode