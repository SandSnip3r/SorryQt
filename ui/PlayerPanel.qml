import QtQuick

Rectangle {
  id: playerPanel
  property var primaryColor: "black"
  property var secondaryColor
  property Repeater cardRepeater: cardRepeater_
  property SequentialAnimation animation: animation_
  property bool isPlaying: false
  color: primaryColor
  border.color: "black"
  border.width: 3
  radius: width * .03
  clip: false

  SequentialAnimation {
    id: animation_
    running: false
    loops: Animation.Infinite
    PropertyAnimation { target: playerPanel; property: "color"; from: primaryColor; to: secondaryColor; duration: 300 }
    PropertyAnimation { target: playerPanel; property: "color"; from: secondaryColor; to: primaryColor; duration: 100 }
    onRunningChanged: {
      if (!animation_.running) {
        playerPanel.color = primaryColor
      }
    }
  }

  Row {
    id: cardRow
    anchors.left: parent.left
    anchors.right: parent.right
    anchors.top: parent.top
    anchors.bottom: parent.bottom
    anchors.leftMargin: parent.width * .04
    anchors.rightMargin: parent.width * .04
    anchors.topMargin: parent.height * .15
    anchors.bottomMargin: parent.height * .15
    spacing: 5
    visible: isPlaying
    Repeater {
      id: cardRepeater_
      model: 5
      Card {
        width: (cardRow.width - cardRow.spacing*4) / 5
        height: cardRow.height*2
        function getRandomCard() {
          var cards = [ "1","2","3","4","5","7","8","10","11","12","Sorry" ]
          return cards[Math.floor(Math.random()*cards.length)]
        }
        cardText: getRandomCard()
      }
    }
  }
}