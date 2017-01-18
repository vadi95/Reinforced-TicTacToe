# Reinforced-TicTacToe
Two Neural Nets learn how to play TicTacToe

<h1> Policy Gradient </h1>
In this approach you play random games (both players make random moves), and say if player 'O' wins you reward all 'O' moves positively (reduced by a discount factor i.e. the final move gets the full reward and then reward decreases by a factor) and reward 'X' moves negatively in the same fashion. If the game results in a draw we can reward both players with a lesser positive reward (again discounted by a factor).<br><br>
You might end up rewarding sub optimal moves positively and vice versa but over a large number of games things work in your favor.

I trained over 50 Million games and gradients were updated every 5000 games. Other than few tricky moves that lead to winning positions (not immidiately winning) it seems to perform well.<br><br>

<h1> Supervised </h1>
Here I wanted to only update the network with the best possible action for a state. Again all moves are completely random (exploratory). For the first 100,000 games the networks learnt immidiate winning moves and learnt to block and avoid an immidiate loss.<br><br>

For 3x3 TicTacToe there are 5478 valid and distinct boards, out of which 958 are terminal states so we don't have to learn actions for those. We can stop training when we have learnt 4520 (5478 - 958) states. It took another 200,000 games to master all the states. To update states without immidiate reward, the players play out the game from that state with their current model and learn from the end reward (if they are sure about the moves played). (This is sort of Q-Learning) <br><br>

This resulted in better models than policy gradient. (Seems like it gets the best possible reward always)
