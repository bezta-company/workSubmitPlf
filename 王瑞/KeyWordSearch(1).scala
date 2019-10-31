package bzt.search

import java.io.File

object KeyWordSearch {

  def main(args: Array[String]): Unit = {
    val result = getMatchDocs("中国")
    result.map(x => println(x))
  }

//  1、统计各个文档中各单词出现了多少次， 得到结果为一个集合，
//  集合的元素类型为 对象(自定义一个class),
//  对象包含两个字段（文档名字， 单词次数的统计情况(Map)）
//  2、统计所有的文档总共包含多少个单词， 得到结果为一个集合A
//  3、将每个文档简单地转换为词频向量，即根据1得到的Map 和 2得到的集合A 进行映射
//  4、将目标文档（即用户输入的一段话）向量化
//  5、计算目标文档向量与 3 得到文档向量的相似度, (用余弦距离)
//  6、挑选相似度最高的前k个文档， 输出其文档名

  case class Match(fileName:String,wordCount:Map[String,Int])
  val set = scala.collection.mutable.HashSet[String]()

  def getMatchDocs(keyWord: String)= {
    val files = new File("data/")
    val keyGroup = files.listFiles().map(file => {
      val words = scala.io.Source.fromFile(file).getLines()
        .flatMap(line => line.split("\\s+"))
        .filter(w => !w.equals(""))
      val wordCount = words.toArray.groupBy(t => t).map(m => (m._1, m._2.length))
      Match(file.getName, wordCount)
    })
    val allWords = keyGroup.map(x =>x.wordCount.map(word => set.add(word._1)))
    set
  }

  //  关键词搜索，map方式以hashmap方式存放
  //  def getMatchTopnDocs(keyWord: String) = {
  //    val docIndex = new mutable.HashMap[String, scala.collection.mutable.Set[String]]()
  //    val files = new File("data/")
  //    for (file <- files.listFiles()) {
  //      val words = scala.io.Source.fromFile(file).getLines()
  //        .flatMap(line => line.split("\\s+"))
  //        .filter(w => !w.equals(""))
  //      for (word <- words) {
  //        //        println(word)
  //        val docs = docIndex.getOrElse(word, new scala.collection.mutable.HashSet[String]())
  //        docs.add(file.getName)
  //        docIndex.put(word, docs)
  //      }
  //    }
  //    docIndex.get(keyWord) match {
  //      case None => Set()
  //      case Some(x) => x
  //    }
  //  }

//  关键词搜索，map方式
//  def getMatchDocs(keyWord: String)= {
    //    val files = new File("data/")
    //    val keyGroup = files.listFiles().flatMap(file => {
    //      val words = scala.io.Source.fromFile(file).getLines()
    //        .flatMap(line => line.split("\\s+"))
    //        .filter(w => !w.equals(""))
    //      words.map(w => (w, file.getName))
    //    })
    //      .groupBy(t => t._1)
    //      .map(t => (t._1,
    //        t._2.map(wd => wd._2)
    //           .groupBy(doc => doc)
    //          .map(x => (x._1, x._2.length))))
    //    val key = keyGroup1.map(x => (x._2.keys,(x._1,x._2.values)))
    //    val data = key.map(x => (x._1,x._2._1)

    //    keyGroup.get(keyWord) match {
    //      case None => Map()            //Seq：空集合
    //      case Some(x) => x     //x.toSeq.sortBy()
    //
    //     }
//}

}
