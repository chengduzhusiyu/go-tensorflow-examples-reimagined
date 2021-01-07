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

	// Load a frozen graph to use for queries
	modelpath := filepath.Join(*modeldir, "mobilenet_v1_1.0_224_frozen.pb")
	model, err := ioutil.ReadFile(modelpath)
	if err != nil {
		log.Fatal(err)
	}

	// Construct an in-memory graph from the serialized form.
	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		log.Fatal(err)
	}

	// Create a session for inference over graph.
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	img, err := imaging.Open(*jpgfile)
	if err != nil {
		log.Fatalf("failed to open image: %v", err)
	}

	height := 224
	width := 224
	resized := imaging.Resize(img, width, height, imaging.Linear)
	imgFloats, err := utils.NormalizeImageHWC(resized, []float32{128, 128, 128}, 128)
	if e