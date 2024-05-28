import QtQuick

Rectangle {
  property var pieceIndex: 0
  height: width
  radius: height/2
  border.color: "black"
  Text {
    text: pieceIndex
    anchors.centerIn: parent
    font.pointSize: parent.height/2
  }
}