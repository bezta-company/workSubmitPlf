package bzt.map_search

import java.io.File

import scala.collection.mutable

object KeyWordSearch {

  def main(args: Array[String]): Unit = {
    print(getMatchDocs("大会党"))
  }


      def getMatchTopnDocs(keyWord: String) = {
        val docIndex = new mutable.HashMap[String, scala.collection.mutable.Set[String]]()
        val files = new File("data/")
        for (file <- files.listFiles()) {
          val words = scala.io.Source.fromFile(file).getLines()
            .flatMap(line => line.split("\\s+"))
            .filter(w => !w.equals(""))
          for (word <- words) {
            //        println(word)
            val docs = docIndex.getOrElse(word, new scala.collection.mutable.HashSet[String]())
            docs.add(file.getName)
            docIndex.put(word, docs)
          }
        }
        docIndex.get(keyWord) match {
          case None => Set()
          case Some(x) => x
        }
      }


//    def getMatchDocs(keyWord: String): Seq[KeyWordSearch.Match] = {
//      val files = new File("data/")
//      val keyGroup = files.listFiles().flatMap(file => {
//        val words = scala.io.Source.fromFile(file).getLines()
//          .flatMap(line => line.split("\\s+"))
//          .filter(w => !w.equals(""))
//        words.map(w => (w, file.getName))
//      })
//        .groupBy(t => t._1)  //groupBy会产生一个hashmap
//        .map(t => (t._1, t._2.map(wd => wd._2).groupBy(d => d).map(w => Match(w._1, w._2.length))))
//      keyGroup.get(keyWord) match {
//        case None => Seq()    //集合
//        case Some(x) => x.toSeq.sortBy(x => x.count)
//      }
//    }
//    case class Match(docName: String, count: Int)



// 1、统计各个文档中各单词出现了多少次， 得到结果为一个集合，
// 集合的元素类型为 对象(自定义一个class), 对象包含两个字段（文档名字， 单词次数的统计情况(Map)）
// 2、统计所有的文档总共包含多少个单词， 得到结果为一个集合A

  def getMatchDocs(keyWord: String): Unit = {
    val files = new File("data/")
    val set = scala.collection.mutable.HashSet[String]()
    val keyGroup = files.listFiles().map(file => {
      val words = scala.io.Source.fromFile(file).getLines()
        .flatMap(line => line.split("\\s+"))
        .filter(w => !w.equals(""))
      val wordCount = words.toArray.groupBy(t => t).map(m => (m._1, m._2.length))
      val answer = new Match(wordCount, file.getName)
      println(answer)
      wordCount.map(wd => set.add(wd._1))
    })
    print(set)
  }
  case class Match(wordCount:Map[String,Int],fileName:String)
}
