#include <Pixy2.h>

Pixy2 pixy;

void setup() 
{
  Serial.begin(115200);

  pixy.init();

}

void loop() {
  
  delay(1000);

  pixy.ccc.getBlocks();

  if (pixy.ccc.numBlocks)
  {
    Serial.print("Detected ");//Ob etwas erkannt wurde
    Serial.println(pixy.ccc.numBlocks);// Anzahl an gefundenen Objecten
    for (int i=0; i<pixy.ccc.numBlocks; i++)
    {
      Serial.print("  object ");
      Serial.print(i);// object nummer x
      Serial.print(": ");
      pixy.ccc.blocks[i].print(); //signature, position, bounding box, index zuerst gefunden, alter
    }
  }
}
