import QtQuick

Rectangle {
  id: card
  property var cardText: ""
  property int highlightCount: 0
  property string cardColor: (highlightCount == 0) ? "#FFFFFF" : "#FFA0FF"
  radius: width/10
  color: cardColor
  Rectangle {
    id: rectyWecty
    property var whiteWidth: 3/80.64 * card.width
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
      anchors.topMargin: card.width * .02
      anchors.left: parent.left
      anchors.leftMargin: card.width * .08
      font.pointSize: Math.max(1, card.width * .2)
    }
  }
}