import org.apache.spark.sql.SparkSession
import sun.nio.cs.ext.DoubleByteEncoder

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object FindSimilarityNames {

  val session = SparkSession.builder().master("local[*]").appName(this.getClass.getName).getOrCreate()

  case class XiaoQu(name: String, price: Int, govId: Int, longitude: Double, latitude: Double, county: String, city: String) {

  }

  def main(args: Array[String]): Unit = {

    val data = session.sparkContext.textFile("data/xiaoqu.csv")

    def toXiaoQu(line :String) = {
      val values = line.split(",")
      try{
        XiaoQu(values(7), values(6).toInt, values(3).toFloat.toInt, values(4).toDouble, values(5).toDouble, values(2), values(1))
      }catch{
        case e :Exception =>
          println("type convert error")
          null
      }

    }

    def getPointDistance(lng1 :Double, lat1 :Double, lng2 :Double, lat2 :Double) = {
      new Random().nextDouble() * 500
    }

    def getSimilarity(text1 :String, text2 :String) = {
      new Random().nextDouble()
    }
    val groupByGov = data.map(line => toXiaoQu(line)).filter(_ !=null)
      .groupBy(xq => xq.govId)
      .map( t => {
        val gov_id= t._1
        val points = t._2.toArray
        val hitInGov = ArrayBuffer[(String, String, Double, Double)]()
        for(i <- 0 until points.length -1){
          val p1 = points(i)

          for(j <- i +1 until points.length){
            val p2 = points(j)
            val dis = getPointDistance(p1.longitude, p1.latitude, p2.longitude, p2.latitude)
            if(dis < 500){
              val sim = getSimilarity(p1.name, p2.name)
              if(sim >= 0.5){
                hitInGov.append((p1.name,  p2.name, dis, sim))
              }
            }
          }
        }
        (gov_id, hitInGov)
      })

    val res = groupByGov.flatMap(t => t._2.map(x => (t._1, x)))

    res.foreach(println)

     

  }
}