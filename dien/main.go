
package main

import (
	"bufio"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
	//Parse flags
	modeldir := flag.String("dir", "./", "Directory containing trained model files. Assumes model file is called frozen_inference_graph.pb")
	//datafile := flag.String()
	flag.Parse()
	if *modeldir == "" {
		flag.Usage()
		return
	}

	modelpath := filepath.Join(*modeldir, "DIEN.pb")
	model, err := ioutil.ReadFile(modelpath)
	if err != nil {
		log.Fatal(err)
	}
	// Construct an in-memory graph from the serialized form.
	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {