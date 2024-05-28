import QtQuick
import QtQuick.Shapes

Rectangle {
  id: slideBar
  height: cellSize/2
  border.color: "black"

  Rectangle {
    height: cellSize*.75
    width: height
    radius: width/2
    anchors.horizontalCenter: parent.left
    anchors.verticalCenter: parent.verticalCenter
    color: slideBar.color
    border.color: "black"
  }

  Shape {
    height: cellSize
    width: cellSize
    anchors.verticalCenter: parent.verticalCenter
    anchors.horizontalCenter: parent.right
    ShapePath {
      fillColor: slideBar.color
      strokeColor: "black"
      startX: cellSize/8
      startY: cellSize/2
      PathLine { x: 7*cellSize/8; y: cellSize/16 }
      PathLine { x: 7*cellSize/8; y: 15*cellSize/16 }
      PathLine { x: cellSize/8; y: cellSize/2 }
    }
  }
}