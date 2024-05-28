import QtQuick
import QtQuick.Controls
import Sorry

Window {
  id: window
  width: 1024
  height: 612
  visible: true
  title: qsTr("Sorry")
  
  Board {
    id: board
    height: parent.height - cardPane.height
    width: height
    anchors.left: parent.left
    anchors.top: parent.top

    readonly property double basePieceSize: 1/32

    Repeater {
      id: pieceRepeater
      model: 4
      Piece {
        width: board.basePieceSize * board.height
        color: "green"
        pieceIndex: index
      }
    }

    function getPos(pieceIndex, boardPos) {
      const pieceSize = board.basePieceSize * board.height
      const leftMargin = board.cellSize/2
      const topMargin = board.cellSize/2
      const cellSize = board.cellSize
      if (boardPos == 0) {
        if (pieceIndex == 0) {
          return [leftMargin + cellSize * 10.5 - pieceSize/2,
                  topMargin + cellSize * 12.9 - pieceSize/2]
        } else if (pieceIndex == 1) {
          return [leftMargin + cellSize * 11.5 - pieceSize/2,
                  topMargin + cellSize * 12.9 - pieceSize/2]
        } else if (pieceIndex == 2) {
          return [leftMargin + cellSize * 10.5 - pieceSize/2,
                  topMargin + cellSize * 13.9 - pieceSize/2]
        } else if (pieceIndex == 3) {
          return [leftMargin + cellSize * 11.5 - pieceSize/2,
                  topMargin + cellSize * 13.9 - pieceSize/2]
        }
      } else if (boardPos == 66) {
        if (pieceIndex == 0) {
          return [leftMargin + cellSize * 12.5 - pieceSize/2,
                  topMargin + cellSize * 8 - pieceSize/2]
        } else if (pieceIndex == 1) {
          return [leftMargin + cellSize * 13.5 - pieceSize/2,
                  topMargin + cellSize * 8 - pieceSize/2]
        } else if (pieceIndex == 2) {
          return [leftMargin + cellSize * 12.5 - pieceSize/2,
                  topMargin + cellSize * 9 - pieceSize/2]
        } else if (pieceIndex == 3) {
          return [leftMargin + cellSize * 13.5 - pieceSize/2,
                  topMargin + cellSize * 9 - pieceSize/2]
        }
      } else if (boardPos >= 1 && boardPos <= 13) {
        // "Start" of bottom row.
        return [leftMargin + cellSize * (13-boardPos) - pieceSize/2,
                topMargin + cellSize * 15 - pieceSize/2]
      } else if (boardPos >= 14 && boardPos <= 28) {
        // Left column.
        return [leftMargin - pieceSize/2,
                topMargin + cellSize * (28-boardPos) - pieceSize/2]
      } else if (boardPos >= 29 && boardPos <= 43) {
        // Top row.
        return [leftMargin + cellSize * (boardPos-28) - pieceSize/2,
                topMargin - pieceSize/2]
      } else if (boardPos >= 44 && boardPos <= 58) {
        // Right column.
        return [leftMargin + cellSize * 15 - pieceSize/2,
                topMargin + cellSize * (boardPos-43) - pieceSize/2]
      } else if (boardPos >= 59 && boardPos <= 60) {
        // "End" of bottom row.
        return [leftMargin + cellSize * (13 + 60-boardPos) - pieceSize/2,
                topMargin + cellSize * 15 - pieceSize/2]
      } else if (boardPos >= 61 && boardPos <= 65) {
        // Safe zone.
        return [leftMargin + cellSize * 13 - pieceSize/2,
                topMargin + cellSize * (10 + 65-boardPos) - pieceSize/2]
      }
      return [0, 0]
    }

    function display() {
      if (board.status != Board.Ready) {
        console.log("Board not yet ready, not going to continue")
        return
      }
      // var pos0 = getPos(0, 66)
      // var pos1 = getPos(1, 66)
      // var pos2 = getPos(2, 66)
      // var pos3 = getPos(3, 66)
      // var piece0 = pieceRepeater.itemAt(0)
      // var piece1 = pieceRepeater.itemAt(1)
      // var piece2 = pieceRepeater.itemAt(2)
      // var piece3 = pieceRepeater.itemAt(3)
      // piece0.x = pos0[0]; piece0.y = pos0[1]
      // piece1.x = pos1[0]; piece1.y = pos1[1]
      // piece2.x = pos2[0]; piece2.y = pos2[1]
      // piece3.x = pos3[0]; piece3.y = pos3[1]

      // Position pieces
      var positions = SorryBackend.getPiecePositions();
      for (var i=0; i<positions.length; ++i) {
        var pos = getPos(i, positions[i])
        var piece = pieceRepeater.itemAt(i)
        if (piece && pos) {
          piece.x = pos[0]
          piece.y = pos[1]
        } else if (!piece) {
          console.log("Piece ", i, " is null")
        } else {
          console.log("Pos is null")
        }
      }

      // Populate cards
      const cardStrings = SorryBackend.getCardStrings()
      for (var i=0; i<5; ++i) {
        var card = cardRepeater.itemAt(i)
        if (card) {
          cardRepeater.itemAt(i).cardText = cardStrings[i]
        } else {
          console.log("Card ", i, " is null")
        }
      }

      moveCountText.moveCount = SorryBackend.getMoveCount()
    }

    // Ensure all dynamic rectangles are destroyed when parentRectangle is destroyed
    Component.onCompleted: display()
    onWidthChanged: {
      display()
    }

    onHeightChanged: {
      display()
    }

    MouseArea {
      anchors.fill: parent
      onClicked: {
        console.log(mouseX, " ", mouseY)
      }
    }

    Canvas {
      id: canvas
      anchors.fill: parent

      property int index1: -1
      property int srcPos1: -1
      property int destPos1: -1

      property int index2: -1
      property int srcPos2: -1
      property int destPos2: -1

      // Code to draw a simple arrow on TypeScript canvas got from https://stackoverflow.com/a/64756256/867349
      function arrow(context, fromx, fromy, tox, toy) {
        const dx = tox - fromx;
        const dy = toy - fromy;
        const headlen = 10; //Math.sqrt(dx * dx + dy * dy) * 0.3; // length of head in pixels
        const angle = Math.atan2(dy, dx);
        context.beginPath();
        context.moveTo(fromx, fromy);
        context.lineTo(tox, toy);
        context.lineWidth = 3.5*(board.width/512)
        context.strokeStyle = "#FF00FF"
        context.stroke();
        context.beginPath();
        context.moveTo(tox - headlen * Math.cos(angle - Math.PI / 6), toy - headlen * Math.sin(angle - Math.PI / 6));
        context.lineTo(tox, toy );
        context.lineTo(tox - headlen * Math.cos(angle + Math.PI / 6), toy - headlen * Math.sin(angle + Math.PI / 6));
        context.stroke();
      }

      function setFirstMove(index, srcPosIndex, destPosIndex) {
        canvas.index1 = index
        canvas.srcPos1 = srcPosIndex
        canvas.destPos1 = destPosIndex
      }

      function setSecondMove(index, srcPosIndex, destPosIndex) {
        canvas.index2 = index
        canvas.srcPos2 = srcPosIndex
        canvas.destPos2 = destPosIndex
      }

      function resetMoves() {
        canvas.index1 = -1
        canvas.srcPos1 = -1
        canvas.destPos1 = -1
        canvas.index2 = -1
        canvas.srcPos2 = -1
        canvas.destPos2 = -1
      }

      onPaint: {
        // Get the canvas context
        var ctx = getContext("2d");
        // Fill a solid color rectangle
        // ctx.fillStyle = Qt.rgba(1, 0.7, 0.1, 0.1);
        ctx.clearRect(0, 0, width, height);

        var adjust = board.basePieceSize * board.height / 2
        if (index1 >= 0 && srcPos1 >= 0 && destPos1 >= 0) {
          var src = board.getPos(index1, srcPos1)
          var dest = board.getPos(index1, destPos1)
          arrow(ctx, src[0]+adjust, src[1]+adjust, dest[0]+adjust, dest[1]+adjust)
        }
        if (index2 >= 0 && srcPos2 >= 0 && destPos2 >= 0) {
          var src = board.getPos(index2, srcPos2)
          var dest = board.getPos(index2, destPos2)
          arrow(ctx, src[0]+adjust, src[1]+adjust, dest[0]+adjust, dest[1]+adjust)
        }

        // Draw an arrow on given context starting at position (0, 0) -- top left corner up to position (mouseX, mouseY)
        //   determined by mouse coordinates position
        // arrow(ctx, 0, 0, ma.mouseX, ma.mouseY)
      }

      MouseArea {
        id: ma
        anchors.fill: parent
        hoverEnabled: true
        // Do a paint requests on each mouse position change (X and Y separately)
        onMouseXChanged: canvas.requestPaint()
        onMouseYChanged: canvas.requestPaint()
      }
    }
  }

  Rectangle {
    id: textPane
    height: 20
    color: "black"
    anchors.top: board.bottom
    anchors.left: board.left
    anchors.right: board.right

    Text {
      property var moveCount: 0
      id: moveCountText
      text: "Moves: " + moveCount
      color: "white"
      anchors.bottom: parent.bottom
      anchors.horizontalCenter: parent.horizontalCenter
    }
  }

  Rectangle {
    id: cardPane
    height: 64
    anchors.top: textPane.bottom
    anchors.left: board.left
    anchors.right: board.right
    color: "black"
    Row {
      id: cardRow
      anchors.left: parent.left
      anchors.right: parent.right
      anchors.top: parent.top
      anchors.bottom: parent.bottom
      anchors.leftMargin: 3
      anchors.rightMargin: 3
      spacing: 5
      Repeater {
        id: cardRepeater
        model: 5
        Card {
          width: (cardRow.width - cardRow.spacing*4) / 5
          height: cardRow.height * 2
          cardText: "12"
        }
      }
    }
  }

  Rectangle {
    anchors.right: parent.right
    anchors.top: parent.top
    anchors.bottom: parent.bottom
    anchors.left: board.right
    color: "grey"

    ListView {
      anchors.fill: parent
      spacing: 1
      model: ListModel {
        id: actionModel
        // Method to sync actions from backend
        function syncActions() {
            clear();
            var actions = SorryBackend.getActions();
            for (var i = 0; i < actions.length; ++i) {
              var action = actions[i]
              append({action})
            }
            board.display()
        }
      }
      delegate: Rectangle {
          id: actionButton
          width: 360
          height: 40
          color: "black"
          border.color: "white"
          Rectangle {
            anchors.left: parent.left
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            width: parent.width * modelData.score
            color: "#00FF00"
            opacity: .2
            radius: parent.radius
            border.color: "transparent"
            border.width: parent.border.width
          }
          Text {
            anchors.centerIn: parent
            // text: index + ": " + modelData.name
            text: modelData.name + ": " + modelData.score.toFixed(2)
            font.pointSize: 16
            color: "white"
          }
          MouseArea {
            anchors.fill: parent
            hoverEnabled: true
            property var cardIndex: 0

            onClicked: {
              // Clear cards
              for (var i=0; i<5; ++i) {
                var card = cardRepeater.itemAt(i)
                card.highlightCount = 0
              }
              // Do action
              SorryBackend.doAction(modelData)
            }

            onEntered: {
              actionButton.color = "#404040"

              // Draw a line from the moved piece's src to dest
              var srcAndDest = SorryBackend.getSrcAndDestPositionsForAction(modelData)
              if (srcAndDest.length >= 3) {
                canvas.setFirstMove(srcAndDest[0], srcAndDest[1], srcAndDest[2])
                if (srcAndDest.length >= 6) {
                  canvas.setSecondMove(srcAndDest[3], srcAndDest[4], srcAndDest[5])
                }
                canvas.requestPaint()
              }
              
              // *Draw a line from the second moved piece's src to dest
              // Highlight the used card
              var cardIndices = SorryBackend.getCardIndicesForAction(modelData)
              for (var i=0; i<cardIndices.length; ++i) {
                var card = cardRepeater.itemAt(cardIndices[i])
                card.highlightCount++
              }
            }

            onExited: {
              actionButton.color = "#000000"

              canvas.resetMoves()
              canvas.requestPaint()

              var cardIndices = SorryBackend.getCardIndicesForAction(modelData)
              for (var i=0; i<cardIndices.length; ++i) {
                var card = cardRepeater.itemAt(cardIndices[i])
                card.highlightCount--
              }
            }
          }
      }
      ScrollIndicator.vertical: ScrollIndicator { }

      Component.onCompleted: actionModel.syncActions()

      Connections {
          target: SorryBackend
          function onActionsChanged() {
            actionModel.syncActions()
          }
      }
    }
  }
}
