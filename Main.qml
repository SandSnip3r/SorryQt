import QtQuick
import QtQuick.Controls
import Sorry

Window {
  id: window
  width: 1400
  height: 912
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

      property var moves: []

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

      function pushMove(move) {
        moves.push(move)
        canvas.requestPaint()
      }

      function popMove() {
        moves.shift()
        canvas.requestPaint()
      }

      onPaint: {
        // Get the canvas context
        var ctx = getContext("2d");
        ctx.clearRect(0, 0, width, height);

        // Make a slight adjustment so that the arrow starts/ends at the center of the piece's position
        var adjust = board.basePieceSize * board.height / 2

        for (const move of moves) {
          var src0 = board.getPos(move.pieceIndex0, move.moveSourcePos0)
          var dest0 = board.getPos(move.pieceIndex0, move.moveDestinationPos0)
          arrow(ctx, src0[0]+adjust, src0[1]+adjust, dest0[0]+adjust, dest0[1]+adjust)
          if ("pieceIndex1" in move) {
            var src1 = board.getPos(move.pieceIndex1, move.moveSourcePos1)
            var dest1 = board.getPos(move.pieceIndex1, move.moveDestinationPos1)
            arrow(ctx, src1[0]+adjust, src1[1]+adjust, dest1[0]+adjust, dest1[1]+adjust)
          }
        }
      }

      Button {
        anchors.verticalCenter: parent.verticalCenter
        anchors.horizontalCenter: parent.horizontalCenter
        width: parent.width * 0.15
        height: parent.height * 0.04
        Text {
          text: "Reset Game"
          font.pointSize: parent.height * .5
          anchors.centerIn: parent
        }
        onClicked: {
          SorryBackend.resetGame()
        }
      }
    }

    Connections {
      target: SorryBackend
      function onBoardStateChanged() {
        board.display()
      }
    }
  }

  Rectangle {
    id: textPane
    height: 30
    color: "black"
    anchors.top: board.bottom
    anchors.left: board.left
    anchors.right: board.right

    Text {
      id: moveCountText
      text: "Moves: " + SorryBackend.moveCount + "   Seed: " + SorryBackend.randomSeed + "   Iterations: " + SorryBackend.iterationCount
      color: "white"
      anchors.horizontalCenter: parent.horizontalCenter
      anchors.verticalCenter: parent.verticalCenter
      font.pointSize: Math.min(parent.width * .05, parent.height * .5)
    }
  }

  Rectangle {
    id: cardPane
    height: 100
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
    id: newActionsPane
    anchors.right: parent.right
    anchors.top: parent.top
    anchors.bottom: parent.bottom
    anchors.left: board.right
    color: "#330000"
    ListView {
      id: actionListView
      anchors.fill: parent
      spacing: 3
      model: SorryBackend.actionListModel
      delegate: Rectangle {
          id: actionButton
          width: newActionsPane.width
          height: width * .1
          color: "black"
          border.color: "white"
          radius: height * .2
          Rectangle {
            // Score "progress" bar
            anchors.left: parent.left
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            width: parent.width * model.score
            color: "#00FF00"
            opacity: .2
            radius: parent.radius
            border.color: "transparent"
            border.width: parent.border.width
          }
          Text {
            anchors.centerIn: parent
            text: model.name + " (" + model.averageMoves.toFixed(1) + " moves)"
            font.pointSize: actionButton.height * .35
            color: "white"
          }
          MouseArea {
            anchors.fill: parent
            hoverEnabled: true

            onClicked: {
              // Clear card highlighting
              for (var i=0; i<5; ++i) {
                var card = cardRepeater.itemAt(i)
                card.highlightCount = 0
              }
              // Do action
              SorryBackend.doAction(model.index)
            }

            onEntered: {
              actionButton.color = "#404040"

              // Draw a line from the moved piece's src to dest (and for the second piece too, if this is a double move)
              var srcAndDest = SorryBackend.getSrcAndDestPositionsForAction(model.index)
              if (srcAndDest.length >= 3) {
                var canvasMove = {}
                canvasMove.pieceIndex0 = srcAndDest[0]
                canvasMove.moveSourcePos0 = srcAndDest[1]
                canvasMove.moveDestinationPos0 = srcAndDest[2]
                if (srcAndDest.length >= 6) {
                  canvasMove.pieceIndex1 = srcAndDest[3]
                  canvasMove.moveSourcePos1 = srcAndDest[4]
                  canvasMove.moveDestinationPos1 = srcAndDest[5]
                }
                canvas.pushMove(canvasMove)
              }

              // Highlight the used card
              var cardIndices = SorryBackend.getCardIndicesForAction(model.index)
              for (var i=0; i<cardIndices.length; ++i) {
                var card = cardRepeater.itemAt(cardIndices[i])
                card.highlightCount++
              }
            }

            onExited: {
              actionButton.color = "#000000"

              canvas.popMove()

              var cardIndices = SorryBackend.getCardIndicesForAction(model.index)
              for (var i=0; i<cardIndices.length; ++i) {
                var card = cardRepeater.itemAt(cardIndices[i])
                card.highlightCount--
              }
            }
          }
      }
    }
  }
}
