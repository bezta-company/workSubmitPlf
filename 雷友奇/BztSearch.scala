package com.BztSearch

import java.io.File

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

object BztSearch {

  def docWordsRecod(): Unit ={
    val files = new File("F:\\doc")
    val keyGroup = files.listFiles().flatMap(file => {
      val words = scala.io.Source.fromFile(file).getLines()
        .flatMap(line => line.split("\\s+"))
        .filter(w => !w.equals(""))
      words.map(w => (w, file.getName))
    })
      .groupBy(each => each._2)
      .map(each => (each._1, each._2.map(each => each._1)))
      .map(each => (each._1, each._2.groupBy(str => str).map(each => (each._1, each._2.length))))

    keyGroup.map(each => new DocWordsRecords(each._1,each._2).showInfo())

  }
  def wordCount():mutable.HashMap[String, Array[String]]={
    val files = new File("F:\\doc")
    val keyGroup = files.listFiles().flatMap(file => {
      val words = scala.io.Source.fromFile(file).getLines()
        .flatMap(line => line.split("\\s+"))
        .filter(w => !w.equals(""))
      words.map(w => (w, file.getName))
    })
      .groupBy(each => each._2)
      .map(each => (each._1, each._2.map(each => each._1).distinct))

    val result = new mutable.HashMap[String,Array[String]]()
    keyGroup.map(each => result.put(each._1,each._2))

    return result
  }

  def main(args: Array[String]):Unit={
    docWordsRecod()
    println()
    wordCount()
  }

}
