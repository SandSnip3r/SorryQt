import QtQuick

Rectangle {
  property var cardText: ""
  property int highlightCount: 0
  property string cardColor: (highlightCount == 0) ? "#FFFFFF" : "#FFFFA0"
  radius: width/10
  color: cardColor
  Rectangle {
    property var whiteWidth: 3
    width: parent.width - whiteWidth*2
    height: parent.height - whiteWidth*2
    anchors.centerIn: parent
    color: cardColor
    border.color: "black"
    border.width: 2
    radius: parent.radius-1
    Text {
      text: cardText
      anchors.top: parent.top
      anchors.topMargin: 5
      anchors.horizontalCenter: parent.horizontalCenter
      font.pointSize: 22
    }
  }
}