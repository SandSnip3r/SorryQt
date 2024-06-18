import QtQuick
import QtQuick.Shapes

Rectangle {
  id: board
  color: "white"

  property real cellSize: board.height / 16.0
  property real homeSize: cellSize*2.2
  readonly property string greenPrimaryColor:    "#44CC44"
  readonly property string greenSecondaryColor:  "#88FF88"
  readonly property string redPrimaryColor:      "#EE3333"
  readonly property string redSecondaryColor:    "#FF8888"
  readonly property string bluePrimaryColor:     "#4444EE"
  readonly property string blueSecondaryColor:   "#8888FF"
  readonly property string yellowPrimaryColor:   "#EEEE22"
  readonly property string yellowSecondaryColor: "#AAAA22"

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
    color: board.greenPrimaryColor
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
        color: board.greenPrimaryColor
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
    color: board.greenPrimaryColor
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
    color: board.yellowPrimaryColor
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
        color: board.yellowPrimaryColor
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
    color: board.yellowPrimaryColor
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
    color: board.bluePrimaryColor
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
        color: board.bluePrimaryColor
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
    color: board.bluePrimaryColor
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
    color: board.redPrimaryColor
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
        color: board.redPrimaryColor
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
    color: board.redPrimaryColor
    anchors.left: redSafetyZone.right
    anchors.leftMargin: -board.cellSize/10
    anchors.verticalCenter: redSafetyZone.verticalCenter
  }

  Slide {
    id: greenSlideShort
    width: cellSize*3
    color: board.greenPrimaryColor
    anchors.left: greenStart.horizontalCenter
    anchors.verticalCenter: bottomRow.verticalCenter
  }

  Slide {
    id: greenSlideLong
    width: cellSize*4
    color: board.greenPrimaryColor
    anchors.right: greenSlideShort.left
    anchors.rightMargin: cellSize*5
    anchors.verticalCenter: bottomRow.verticalCenter
  }

  Slide {
    id: redSlideShort
    width: cellSize*3
    color: board.redPrimaryColor
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
    color: board.redPrimaryColor
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
    color: board.bluePrimaryColor
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
    color: board.bluePrimaryColor
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
    color: board.yellowPrimaryColor
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
    color: board.yellowPrimaryColor
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