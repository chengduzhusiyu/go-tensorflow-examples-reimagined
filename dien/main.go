
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
		log.Fatal(err)
	}

	// Create a session for inference over graph.
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	source := Dataprocess(16)
	uids, mids, cats, mid_his, cat_his, mid_mask, sl := prepare_data(source)

	// uids, mids, cats, mid_his, cat_his, mid_mask, length_x
	inputOp0 := graph.Operation("Inputs/mid_his_batch_ph")
	inputOp1 := graph.Operation("Inputs/cat_his_batch_ph")
	inputOp2 := graph.Operation("Inputs/uid_batch_ph")
	inputOp3 := graph.Operation("Inputs/mid_batch_ph")
	inputOp4 := graph.Operation("Inputs/cat_batch_ph")
	inputOp5 := graph.Operation("Inputs/mask")
	inputOp6 := graph.Operation("Inputs/seq_len_ph")

	// inputOp7 := graph.Operation("Inputs/noclk_mid_batch_ph")
	// inputOp8 := graph.Operation("Inputs/noclk_cat_batch_ph")
	// inputOp9 := graph.Operation("Inputs/target_ph")

	o1 := graph.Operation("dien/fcn/Softmax")
	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			inputOp0.Output(0): mid_his,
			inputOp1.Output(0): cat_his,
			inputOp2.Output(0): uids,
			inputOp3.Output(0): mids,
			inputOp4.Output(0): cats,
			inputOp5.Output(0): mid_mask,
			inputOp6.Output(0): sl,
			// inputOp7.Output(0): noClkMidHis,
			// inputOp8.Output(0): noClkCatHis,
			// inputOp9.Output(0): target,
		},
		[]tf.Output{
			o1.Output(0),
		},
		nil)