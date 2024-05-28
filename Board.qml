import QtQuick
import QtQuick.Shapes

Rectangle {
  id: board
  color: "white"

  property real cellSize: board.height / 16.0
  property real homeSize: cellSize*2.2
  readonly property string greenColor: "#44CC44"
  readonly property string redColor: "#EE3333"
  readonly property string blueColor: "#4444EE"
  readonly property string yellowColor: "#EEEE22"

  Row {
    id: topRow
    anchors.top: parent.top
    anchors.left: parent.left
    Repeater {
      model: 16
      Rectangle {
        width: board.cellSize
        height: board.cellSize
        color: "white"
        border.color: "black"
      }
    }
  }

  Row {
    id: bottomRow
    anchors.bottom: parent.bottom
    anchors.left: parent.left
    Repeater {
      model: 16
      Rectangle {
        width: board.cellSize
        height: board.cellSize
        color: "white"
        border.color: "black"
      }
    }
  }

  Column {
    id: leftColumn
    anchors.top: topRow.bottom
    anchors.bottom: bottomRow.top
    anchors.left: parent.left
    Repeater {
      model: 14
      Rectangle {
        width: board.cellSize
        height: board.cellSize
        color: "white"
        border.color: "black"
      }
    }
  }

  Column {
    id: rightColumn
    anchors.top: topRow.bottom
    anchors.bottom: bottomRow.top
    anchors.right: parent.right
    Repeater {
      model: 14
      Rectangle {
        width: board.cellSize
        height: board.cellSize
        color: "white"
        border.color: "black"
      }
    }
  }

  // Green

  Rectangle {
    id: greenStart
    width: board.homeSize
    height: width
    x: 11.5*board.cellSize - width/2
    y: board.height - board.cellSize - height
    radius: width/2
    border.color: "black"
    color: board.greenColor
  }

  Column {
    id: greenSafetyZone
    anchors.bottom: bottomRow.top
    x: 13.5*board.cellSize - width/2
    Repeater {
      model: 5
      Rectangle {
        width: board.cellSize
        height: board.cellSize
        color: board.greenColor
        border.color: "black"
      }
    }
  }

  Rectangle {
    id: greenHome
    width: board.homeSize
    height: width
    radius: width/2
    border.color: "black"
    color: board.greenColor
    anchors.bottom: greenSafetyZone.top
    anchors.bottomMargin: -board.cellSize/10
    anchors.horizontalCenter: greenSafetyZone.horizontalCenter
  }

  // Yellow

  Rectangle {
    id: yellowStart
    width: board.homeSize
    height: width
    x: board.width - board.cellSize - width
    y: 4.5*board.cellSize - height/2
    radius: width/2
    border.color: "black"
    color: board.yellowColor
  }

  Row {
    id: yellowSafetyZone
    anchors.right: rightColumn.left
    y: 2.5*board.cellSize - height/2
    Repeater {
      model: 5
      Rectangle {
        width: board.cellSize
        height: board.cellSize
        color: board.yellowColor
        border.color: "black"
      }
    }
  }

  Rectangle {
    id: yellowHome
    width: board.homeSize
    height: width
    radius: width/2
    border.color: "black"
    color: board.yellowColor
    anchors.right: yellowSafetyZone.left
    anchors.rightMargin: -board.cellSize/10
    anchors.verticalCenter: yellowSafetyZone.verticalCenter
  }

  // Blue

  Rectangle {
    id: blueStart
    width: board.homeSize
    height: width
    x: 4.5*board.cellSize - width/2
    y: board.cellSize
    radius: width/2
    border.color: "black"
    color: board.blueColor
  }

  Column {
    id: blueSafetyZone
    anchors.top: topRow.bottom
    x: 2.5*board.cellSize - width/2
    Repeater {
      model: 5
      Rectangle {
        width: board.cellSize
        height: board.cellSize
        color: board.blueColor
        border.color: "black"
      }
    }
  }

  Rectangle {
    id: blueHome
    width: board.homeSize
    height: width
    radius: width/2
    border.color: "black"
    color: board.blueColor
    anchors.top: blueSafetyZone.bottom
    anchors.topMargin: -board.cellSize/10
    anchors.horizontalCenter: blueSafetyZone.horizontalCenter
  }

  // Red

  Rectangle {
    id: redStart
    width: board.homeSize
    height: width
    x: board.cellSize
    y: 11.5*board.cellSize - height/2
    radius: width/2
    border.color: "black"
    color: board.redColor
  }

  Row {
    id: redSafetyZone
    anchors.left: leftColumn.right
    y: 13.5*board.cellSize - height/2
    Repeater {
      model: 5
      Rectangle {
        width: board.cellSize
        height: board.cellSize
        color: board.redColor
        border.color: "black"
      }
    }
  }

  Rectangle {
    id: redHome
    width: board.homeSize
    height: width
    radius: width/2
    border.color: "black"
    color: board.redColor
    anchors.left: redSafetyZone.right
    anchors.leftMargin: -board.cellSize/10
    anchors.verticalCenter: redSafetyZone.verticalCenter
  }

  Slide {
    id: greenSlideShort
    width: cellSize*3
    color: board.greenColor
    anchors.left: greenStart.horizontalCenter
    anchors.verticalCenter: bottomRow.verticalCenter
  }

  Slide {
    id: greenSlideLong
    width: cellSize*4
    color: board.greenColor
    anchors.right: greenSlideShort.left
    anchors.rightMargin: cellSize*5
    anchors.verticalCenter: bottomRow.verticalCenter
  }

  Slide {
    id: redSlideShort
    width: cellSize*3
    color: board.redColor
    transform: Rotation {
      origin.x: 0
      origin.y: redSlideShort.height/2
      angle: 90
    }
    anchors.verticalCenter: redStart.verticalCenter
    anchors.left: parent.left
    anchors.leftMargin: cellSize/2
  }

  Slide {
    id: redSlideLong
    width: cellSize*4
    color: board.redColor
    transform: Rotation {
      origin.x: 0
      origin.y: redSlideLong.height/2
      angle: 90
    }
    anchors.top: parent.top
    anchors.topMargin: cellSize * 2.5 - height/2
    anchors.left: parent.left
    anchors.leftMargin: cellSize/2
  }

  Slide {
    id: blueSlideShort
    width: cellSize*3
    color: board.blueColor
    anchors.left: blueStart.horizontalCenter
    anchors.verticalCenter: topRow.verticalCenter
    transform: Rotation {
      origin.x: 0
      origin.y: blueSlideShort.height/2
      angle: 180
    }
  }

  Slide {
    id: blueSlideLong
    width: cellSize*4
    color: board.blueColor
    anchors.left: parent.left
    anchors.leftMargin: cellSize * 13.5
    anchors.verticalCenter: topRow.verticalCenter
    transform: Rotation {
      origin.x: 0
      origin.y: blueSlideLong.height/2
      angle: 180
    }
  }

  Slide {
    id: yellowSlideShort
    width: cellSize*3
    color: board.yellowColor
    transform: Rotation {
      origin.x: 0
      origin.y: yellowSlideShort.height/2
      angle: 270
    }
    anchors.verticalCenter: yellowStart.verticalCenter
    anchors.left: parent.right
    anchors.leftMargin: -cellSize/2
  }

  Slide {
    id: yellowSlideLong
    width: cellSize*4
    color: board.yellowColor
    transform: Rotation {
      origin.x: 0
      origin.y: yellowSlideLong.height/2
      angle: 270
    }
    anchors.top: parent.top
    anchors.topMargin: cellSize * 13.5 - height/2
    anchors.left: parent.right
    anchors.leftMargin: -cellSize/2
  }
}