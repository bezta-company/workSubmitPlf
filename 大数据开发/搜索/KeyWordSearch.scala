package bzt.search

import java.io.File

import scala.collection.mutable

object KeyWordSearch {

  def main(args: Array[String]): Unit = {
    println("kkkk")
  }


  //
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

  def getMatchDocs(keyWord: String): Seq[String] = {
    val files = new File("data/")
    val keyGroup = files.listFiles().flatMap(file => {
      val words = scala.io.Source.fromFile(file).getLines()
        .flatMap(line => line.split("\\s+"))
          .filter(w => !w.equals(""))
      words.map(w => (w, file.getName))
    })
      .groupBy(t => t._1)
      .map(t => (t._1, t._2.map(each => each._2).distinct))
    keyGroup.get(keyWord) match {
      case None => Seq()
      case Some(x) => x
    }


    def getMatchDocsBySimilarity(keyWord: String): Seq[String] = {
      val files = new File("data/")
      val keyGroup = files.listFiles().map(file => {
        val words = scala.io.Source.fromFile(file).getLines()
          .flatMap(line => line.split("\\s+"))
          .filter(w => !w.equals(""))
        val wordsCount = words.toArray.groupBy(w=>w).map(wc=> (wc._1, wc._2.length))
        myclass(wordsCount, file.getName)
      })


  }
  
  //  def getMatchDocs(keyWord: String): Seq[String] = {
//    val files = new File("data/")
//    val keyGroup = files.listFiles().flatMap(file => {
//      val words = scala.io.Source.fromFile(file).getLines()
//        .flatMap(line => line.split("\\s+"))
//        .filter(w => !w.equals(""))
//      words.map(w => (w, file.getName))
//    })
//      .groupBy(t => t._1)
//      .map(t => (t._1,
//        t._2.map(wd => wd._2) // Array[String] 得到单词在哪些文档出现了
//          .groupBy(d => d) // 得到文档出现 的次数
////          .map(dc => (dc._1, dc._2.length))
//          .map(dc => MatchMeta(dc._1, dc._2.length))// 改为case class
//      ))
//    // Map(关键词， 对象集合( matchMeata(文档名字，次数))
//    keyGroup.get(keyWord) match {
//      case None => Seq()
//      case Some(x) => x.toSeq.sortBy()//....自行补充
//    }
//  }

  case class MatchMeta(docName: String, count: Int)

    def getMatchTopnDocs3(keyWord: String, n :Int) :Seq[MatchMeta] = {
      val files = new File("data/")
      val keyGroup = files.listFiles().flatMap(file =>{
        val words = scala.io.Source.fromFile(file).getLines()
          .flatMap(line => line.split("\\s+"))
          .filter(w => !w.equals(""))
        words.map(w => (w, file.getName))
      })
        .groupBy(t => t._1)
        .map(t => (t._1, t._2.map(wd => wd._2)
          .groupBy(d => d)
          .map(dc => MatchMeta(dc._1, dc._2.length))
        ))

      keyGroup.get(keyWord) match {
        case None => Seq()
        case Some(x) => x.toSeq.sortBy(m => m.count)(Ordering.Int.reverse).take(n)
      }
    }


}
