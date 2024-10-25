import QtQuick
import QtQuick.Controls
import Sorry

Window {
  id: window
  width: 1300
  height: 1100
  visible: true
  title: qsTr("Sorry")
  minimumWidth: board.minHeight + 200
  minimumHeight: board.minHeight + textPane.height + playerPanels.height

  Board {
    id: board
    readonly property double minHeight: 500
    height: Math.max(minHeight, parent.height - playerPanels.height - textPane.height)
    width: height
    anchors.left: parent.left
    anchors.top: parent.top

    readonly property double basePieceSize: 1/32

    function getPieceRepeater(playerColor) {
      if (playerColor == PlayerColor.Green) {
        return greenPieceRepeater
      } else if (playerColor == PlayerColor.Red) {
        return redPieceRepeater
      } else if (playerColor == PlayerColor.Blue) {
        return bluePieceRepeater
      } else if (playerColor == PlayerColor.Yellow) {
        return yellowPieceRepeater
      } else {
        console.log("Unknown player; cannot get piece repeater")
        return null
      }
    }

    Repeater {
      id: greenPieceRepeater
      model: 4
      Piece {
        width: board.basePieceSize * board.height
        color: board.greenSecondaryColor
        pieceIndex: index
        visible: false
      }
    }

    Repeater {
      id: redPieceRepeater
      model: 4
      Piece {
        width: board.basePieceSize * board.height
        color: board.redSecondaryColor
        pieceIndex: index
        visible: false
      }
    }

    Repeater {
      id: bluePieceRepeater
      model: 4
      Piece {
        width: board.basePieceSize * board.height
        color: board.blueSecondaryColor
        pieceIndex: index
        visible: false
      }
    }

    Repeater {
      id: yellowPieceRepeater
      model: 4
      Piece {
        width: board.basePieceSize * board.height
        color: board.yellowSecondaryColor
        pieceIndex: index
        visible: false
      }
    }

    function getPos(player, pieceIndex, boardPos) {
      const pieceSize = board.basePieceSize * board.height
      const leftMargin = board.cellSize/2
      const topMargin = board.cellSize/2
      const cellSize = board.cellSize
      if (boardPos == 0) {
        if (player == PlayerColor.Green) {
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
        } else if (player == PlayerColor.Red) {
          if (pieceIndex == 0) {
            return [leftMargin + cellSize * 2.125 - pieceSize/2,
                    topMargin + cellSize * 10.5 - pieceSize/2]
          } else if (pieceIndex == 1) {
            return [leftMargin + cellSize * 2.125 - pieceSize/2,
                    topMargin + cellSize * 11.5 - pieceSize/2]
          } else if (pieceIndex == 2) {
            return [leftMargin + cellSize * 1.125 - pieceSize/2,
                    topMargin + cellSize * 10.5 - pieceSize/2]
          } else if (pieceIndex == 3) {
            return [leftMargin + cellSize * 1.125 - pieceSize/2,
                    topMargin + cellSize * 11.5 - pieceSize/2]
          }
        } else if (player == PlayerColor.Blue) {
          if (pieceIndex == 0) {
            return [leftMargin + cellSize * 4.5 - pieceSize/2,
                    topMargin + cellSize * 2.1 - pieceSize/2]
          } else if (pieceIndex == 1) {
            return [leftMargin + cellSize * 3.5 - pieceSize/2,
                    topMargin + cellSize * 2.1 - pieceSize/2]
          } else if (pieceIndex == 2) {
            return [leftMargin + cellSize * 4.5 - pieceSize/2,
                    topMargin + cellSize * 1.1 - pieceSize/2]
          } else if (pieceIndex == 3) {
            return [leftMargin + cellSize * 3.5 - pieceSize/2,
                    topMargin + cellSize * 1.1 - pieceSize/2]
          }
        } else if (player == PlayerColor.Yellow) {
          if (pieceIndex == 0) {
            return [leftMargin + cellSize * 12.875 - pieceSize/2,
                    topMargin + cellSize * 4.5 - pieceSize/2]
          } else if (pieceIndex == 1) {
            return [leftMargin + cellSize * 12.875 - pieceSize/2,
                    topMargin + cellSize * 3.5 - pieceSize/2]
          } else if (pieceIndex == 2) {
            return [leftMargin + cellSize * 13.875 - pieceSize/2,
                    topMargin + cellSize * 4.5 - pieceSize/2]
          } else if (pieceIndex == 3) {
            return [leftMargin + cellSize * 13.875 - pieceSize/2,
                    topMargin + cellSize * 3.5 - pieceSize/2]
          }
        }
      } else if (boardPos == 66) {
        if (player == PlayerColor.Green) {
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
        } else if (player == PlayerColor.Red) {
          if (pieceIndex == 0) {
            return [leftMargin + cellSize * 7 - pieceSize/2,
                    topMargin + cellSize * 12.5 - pieceSize/2]
          } else if (pieceIndex == 1) {
            return [leftMargin + cellSize * 7 - pieceSize/2,
                    topMargin + cellSize * 13.5 - pieceSize/2]
          } else if (pieceIndex == 2) {
            return [leftMargin + cellSize * 6 - pieceSize/2,
                    topMargin + cellSize * 12.5 - pieceSize/2]
          } else if (pieceIndex == 3) {
            return [leftMargin + cellSize * 6 - pieceSize/2,
                    topMargin + cellSize * 13.5 - pieceSize/2]
          }
        } else if (player == PlayerColor.Blue) {
          if (pieceIndex == 0) {
            return [leftMargin + cellSize * 2.5 - pieceSize/2,
                    topMargin + cellSize * 7 - pieceSize/2]
          } else if (pieceIndex == 1) {
            return [leftMargin + cellSize * 1.5 - pieceSize/2,
                    topMargin + cellSize * 7 - pieceSize/2]
          } else if (pieceIndex == 2) {
            return [leftMargin + cellSize * 2.5 - pieceSize/2,
                    topMargin + cellSize * 6 - pieceSize/2]
          } else if (pieceIndex == 3) {
            return [leftMargin + cellSize * 1.5 - pieceSize/2,
                    topMargin + cellSize * 6 - pieceSize/2]
          }
        } else if (player == PlayerColor.Yellow) {
          if (pieceIndex == 0) {
            return [leftMargin + cellSize * 8 - pieceSize/2,
                    topMargin + cellSize * 2.5 - pieceSize/2]
          } else if (pieceIndex == 1) {
            return [leftMargin + cellSize * 8 - pieceSize/2,
                    topMargin + cellSize * 1.5 - pieceSize/2]
          } else if (pieceIndex == 2) {
            return [leftMargin + cellSize * 9 - pieceSize/2,
                    topMargin + cellSize * 2.5 - pieceSize/2]
          } else if (pieceIndex == 3) {
            return [leftMargin + cellSize * 9 - pieceSize/2,
                    topMargin + cellSize * 1.5 - pieceSize/2]
          }
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
        if (player == PlayerColor.Green) {
          return [leftMargin + cellSize * 13 - pieceSize/2,
                  topMargin + cellSize * (10 + 65-boardPos) - pieceSize/2]
        } else if (player == PlayerColor.Red) {
          return [leftMargin + cellSize * (5 + boardPos-65) - pieceSize/2,
                  topMargin + cellSize * 13 - pieceSize/2]
        } else if (player == PlayerColor.Blue) {
          return [leftMargin + cellSize * 2 - pieceSize/2,
                  topMargin + cellSize * (5 + boardPos-65) - pieceSize/2]
        } else if (player == PlayerColor.Yellow) {
          return [leftMargin + cellSize * (10 + 65-boardPos) - pieceSize/2,
                  topMargin + cellSize * 2 - pieceSize/2]
        }
      }
      console.log("Asking for invalid piece position")
      return [0, 0]
    }

    function initialize() {
      // Show everything for active players
      let players = SorryBackend.getPlayers()
      for (let playerColor of players) {
        // Show cards
        let panel = playerPanels.getPlayerPanel(playerColor)
        panel.isPlaying = true

        // Show pieces
        let pieceRepeater = getPieceRepeater(playerColor)
        for (let i=0; i<pieceRepeater.count; ++i) {
          pieceRepeater.itemAt(i).visible = true
        }
      }
    }

    function display() {
      if (board.status != Board.Ready) {
        console.log("Board not yet ready, not going to continue")
        return
      }
      // var pos0 = getPos(PlayerColor.Yellow, 0, 62)
      // var pos1 = getPos(PlayerColor.Yellow, 1, 63)
      // var pos2 = getPos(PlayerColor.Yellow, 2, 64)
      // var pos3 = getPos(PlayerColor.Yellow, 3, 65)
      // var piece0 = yellowPieceRepeater.itemAt(0)
      // var piece1 = yellowPieceRepeater.itemAt(1)
      // var piece2 = yellowPieceRepeater.itemAt(2)
      // var piece3 = yellowPieceRepeater.itemAt(3)
      // piece0.x = pos0[0]; piece0.y = pos0[1]
      // piece1.x = pos1[0]; piece1.y = pos1[1]
      // piece2.x = pos2[0]; piece2.y = pos2[1]
      // piece3.x = pos3[0]; piece3.y = pos3[1]
      // return

      var players = SorryBackend.getPlayers()
      for (var player of players) {

        // Position pieces
        var positions = SorryBackend.getPiecePositionsForPlayer(player)
        var pieceRepeater = getPieceRepeater(player)
        if (pieceRepeater === null) {
          continue
        }
        for (var i=0; i<positions.length; ++i) {
          var piece = pieceRepeater.itemAt(i)
          if (piece) {
            var pos = getPos(player, i, positions[i])
            if (pos) {
              piece.x = pos[0]
              piece.y = pos[1]
            } else {
              console.log("Pos is null")
            }
          }
        }

        // Populate cards
        const cardStrings = SorryBackend.getCardStringsForPlayer(player)
        var playerPanel = playerPanels.getPlayerPanel(player)
        for (var i=0; i<playerPanel.cardRepeater.count; ++i) {
          var card = playerPanel.cardRepeater.itemAt(i)
          if (card) {
            // Reset card highlighting
            card.highlightCount = 0
            card.cardText = cardStrings[i]
          } else {
            console.log("Card is null")
          }
        }
      }
    }

    // Ensure all dynamic rectangles are destroyed when parentRectangle is destroyed
    Component.onCompleted: {
      initialize()
      display()
    }
    onWidthChanged: {
      display()
    }

    onHeightChanged: {
      display()
    }

    Canvas {
      id: canvas
      anchors.fill: parent

      property var moveGroups: ({})

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

      function pushMoveGroup(index, moveGroup) {
        moveGroups[index] = moveGroup
        canvas.requestPaint()
      }

      function popMoveGroup(index) {
        delete moveGroups[index]
        canvas.requestPaint()
      }

      function resetMoveGroups() {
        moveGroups = {}
        canvas.requestPaint()
      }

      onPaint: {
        // Get the canvas context
        var ctx = getContext("2d");
        ctx.clearRect(0, 0, width, height);

        // Make a slight adjustment so that the arrow starts/ends at the center of the piece's position
        var adjust = board.basePieceSize * board.height / 2

        for (const index in moveGroups) {
          for (const move of moveGroups[index]) {
            var src0 = board.getPos(move.playerColor, move.pieceIndex, move.srcPosition)
            var dest0 = board.getPos(move.playerColor, move.pieceIndex, move.destPosition)
            arrow(ctx, src0[0]+adjust, src0[1]+adjust, dest0[0]+adjust, dest0[1]+adjust)
          }
        }
      }
    }

    Connections {
      target: SorryBackend
      function onBoardStateChanged() {
        canvas.resetMoveGroups()
        board.display()
      }

      function onWinRatesChanged(winRates) {
        winRatesText.setWinRates(winRates)
      }
      function onPlayerTurnChanged() {
        canvas.resetMoveGroups()
      }
    }

    Button {
      id: resetButton
      anchors.verticalCenter: parent.verticalCenter
      anchors.horizontalCenter: parent.horizontalCenter
      width: parent.width * 0.15
      height: parent.height * 0.04
      Text {
        text: "Reset Game"
        font.pointSize: Math.max(1, parent.height * .5)
        anchors.centerIn: parent
      }
      onClicked: {
        SorryBackend.resetGame()
      }
    }

    Text {
      id: winRatesText
      anchors.top: resetButton.bottom
      anchors.horizontalCenter: board.horizontalCenter
      text: ""
      color: "black"
      font.pointSize: 12
      function setWinRates(winRates) {
        winRatesText.text = "Green:" + (winRates[0]*100).toFixed(2) + "%\n" +
                            "Red:" + (winRates[1]*100).toFixed(2) + "%\n" +
                            "Blue:" + (winRates[2]*100).toFixed(2) + "%\n" +
                            "Yellow:" + (winRates[3]*100).toFixed(2) + "%"
      }
    }
    Row {
      id: deckCardRow
      anchors.bottom: resetButton.top
      anchors.bottomMargin: board.height * 0.01
      anchors.left: resetButton.left
      anchors.leftMargin: -board.width * 0.02
      property var deckCardWidth: board.width * 0.05
      spacing: -deckCardWidth * 0.9
      Repeater {
        model: SorryBackend.faceDownCardsCount
        Card {
          id: deckCard
          width: deckCardRow.deckCardWidth
          height: deckCardRow.deckCardWidth * 7/4
          cardText: ""
          Text {
            anchors.centerIn: parent
            text: SorryBackend.faceDownCardsCount
            font.pointSize: Math.max(1, deckCardRow.deckCardWidth * .5)
          }
        }
      }
    }

    Rectangle {
      id: winnerRect
      anchors.top: resetButton.bottom
      anchors.horizontalCenter: board.horizontalCenter
      width: board.width * 0.35
      height: width / 2
      radius: width * .03
      border.color: "black"
      border.width: width * 0.02
      color: "#FFBB00"
      visible: SorryBackend.playerTurn == PlayerColor.GameOver
      Text {
        anchors.centerIn: parent
        text: SorryBackend.winner + " wins!"
        font.pointSize: Math.max(1, winnerRect.height * 0.2)
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
      text: "Seed: " + SorryBackend.randomSeed + "   Iterations: " + SorryBackend.iterationCount
      color: "white"
      anchors.horizontalCenter: parent.horizontalCenter
      anchors.verticalCenter: parent.verticalCenter
      font.pointSize: Math.max(1, Math.min(parent.width * .05, parent.height * .5))
    }
  }

  Rectangle {
    color: "black"
    height: 150
    anchors.top: textPane.bottom
    anchors.left: board.left
    anchors.right: board.right
    Grid {
      id: playerPanels
      anchors.fill: parent
      columns: 2

      PlayerPanel {
        height: parent.height/2
        width: parent.width/2
        primaryColor: board.bluePrimaryColor
        secondaryColor: board.blueSecondaryColor
        animation.running: (SorryBackend.playerTurn == PlayerColor.Blue)
      }

      PlayerPanel {
        height: parent.height/2
        width: parent.width/2
        primaryColor: board.yellowPrimaryColor
        secondaryColor: board.yellowSecondaryColor
        animation.running: (SorryBackend.playerTurn == PlayerColor.Yellow)
      }

      PlayerPanel {
        height: parent.height/2
        width: parent.width/2
        primaryColor: board.redPrimaryColor
        secondaryColor: board.redSecondaryColor
        animation.running: (SorryBackend.playerTurn == PlayerColor.Red)
      }

      PlayerPanel {
        height: parent.height/2
        width: parent.width/2
        primaryColor: board.greenPrimaryColor
        secondaryColor: board.greenSecondaryColor
        animation.running: (SorryBackend.playerTurn == PlayerColor.Green)
      }

      function getPlayerPanel(playerColor) {
        if (playerColor === PlayerColor.Green) {
          return playerPanels.children[3]
        } else if (playerColor === PlayerColor.Red) {
          return playerPanels.children[2]
        } else if (playerColor === PlayerColor.Blue) {
          return playerPanels.children[0]
        } else if (playerColor === PlayerColor.Yellow) {
          return playerPanels.children[1]
        }
      }
    }
  }

  Rectangle {
    id: actionsPane
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
          width: actionsPane.width
          height: width * .1
          color: "black"
          border.color: "white"
          radius: height * .2
          property var isHuman: SorryBackend.playerType == PlayerType.Human
          property var isOnlyItem: actionListView.count == 1
          property var previousIsBest: false
          property var isBest: model.isBest
          function drawActionArrows() {
            if (model.index >= 0) {
              // Draw a line from the moved piece's src to dest (and for the second piece too, if this is a double move)
              var moves = SorryBackend.getMovesForAction(model.index)
              canvas.pushMoveGroup(model.index, moves)
            }
          }

          function undrawActionArrows() {
            if (model.index >= 0) {
              canvas.popMoveGroup(model.index)
            }
          }
          onIsBestChanged: {
            if (SorryBackend.playerType == PlayerType.Mcts) {
              if (isBest) {
                drawActionArrows()
              } else {
                undrawActionArrows()
              }
            }
          }
          Rectangle {
            // Score "progress" bar
            anchors.left: parent.left
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            width: parent.width * (isHuman ? 0 : model.score)
            color: "#00FFFF"
            opacity: .2
            radius: parent.radius
            border.color: "transparent"
            border.width: parent.border.width
          }
          Text {
            anchors.centerIn: parent
            text: model.name + ((isHuman || isOnlyItem) ? "" : " (" + (model.score*100).toFixed(2) + "%)")
            font.pointSize: Math.max(1,actionButton.height * .35)
            color: (!isHuman && (isOnlyItem || isBest)) ? "#FFBBFF" : "white"
          }
          MouseArea {
            anchors.fill: parent
            hoverEnabled: true

            onClicked: {
              if (SorryBackend.playerType == PlayerType.Mcts) {
                // Cannot take action for bot
                return
              }
              // Do action
              SorryBackend.doActionFromActionList(model.index)
            }

            onEntered: {
              actionButton.color = "#404040"

              if (model.index >= 0) {
                // Highlight the used card
                var cardIndices = SorryBackend.getCardIndicesForAction(model.index)
                var playerColor = SorryBackend.getPlayerForAction(model.index)
                var playerPanel = playerPanels.getPlayerPanel(playerColor)
                for (let cardIndex of cardIndices) {
                  var card = playerPanel.cardRepeater.itemAt(cardIndex)
                  card.highlightCount++
                }
              }

              if (SorryBackend.playerType == PlayerType.Mcts) {
                // Do not draw actions for MCTS when action entered
                return
              }

              drawActionArrows()
            }

            onExited: {
              actionButton.color = "#000000"

              if (model.index >= 0) {
                var cardIndices = SorryBackend.getCardIndicesForAction(model.index)
                var playerColor = SorryBackend.getPlayerForAction(model.index)
                var playerPanel = playerPanels.getPlayerPanel(playerColor)
                for (let cardIndex of cardIndices) {
                  var card = playerPanel.cardRepeater.itemAt(cardIndex)
                  card.highlightCount--
                }
              }

              if (SorryBackend.playerType == PlayerType.Mcts) {
                // Do not draw actions for MCTS when action entered
                return
              }

              undrawActionArrows()
            }
          }
      }
    }
  }
}
