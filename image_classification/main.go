package main

import (
	"flag"
	"io/ioutil"
	"log"
	"path/filepath"
	"sort"

	"github.com/disintegration/imaging"
	"github.com/k0kubun/pp"
	utils "github.com/rai-project/tensorflow-go-examples"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
	// Parse flags
	modeldir := flag.String("dir", "", "Directory containing trained model files. Assumes model file is called frozen_inference_graph.pb")
	jpgfile := flag.String("jpg", "platypus.jpg", "Path of a JPG image to use for input")
	labelfile := flag.String("labels", "synset1.txt", "Path to file of COCO labels, one per line")
	flag.Parse()
	if *modeldir == "" || *jpgfile == "" {
		flag.Usage()
		return
	}

	// Load the labels
	labels := utils.LoadLabels(*labelfile)

	// Load a frozen graph to use f