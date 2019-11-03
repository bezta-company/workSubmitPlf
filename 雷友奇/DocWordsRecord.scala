package com.BztSearch



class DocWordsRecords(name:String,map:Map[String,Int]){
  def showInfo(): Unit ={
    println(this.name+":")
    for(i <- map.keys){
      print(i+":"+map.get(i).get+"   ")
    }
    println()
  }

}
