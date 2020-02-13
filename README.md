# Marble Ball Algorithm

## Introduction
This is a self-invented algorithm to validate the connectivity of a connected region. Simulate a marble ball bouncing in an irregular connected region of a 2D image and sampled the positions of the ball in equal intervals of time. In this way, a time dimension can be added to the static image, combined with clustering methods, to finally divide the sub-regions of the irregular connected region. 

## Demonstration

1. Input Connected Area.

      ![](https://github.com/RiverLeeGitHub/Marble-Ball-Algorithm/blob/master/Demonstration/img.png?raw=true)


2. Initiate a marble (in green dot), then let it bounce in the domain. Red dots represent the recorded positions in equal intervals of time.

      ![](https://github.com/RiverLeeGitHub/Marble-Ball-Algorithm/blob/master/Demonstration/img_track.png?raw=true)

3. Scenograph of these positions with adding a time dimension. Green dots represent the average positions of recorded positions in a fixed time interval. Blue stars represent the cluster centers of green dots via Mean Shift.

      ![](https://github.com/RiverLeeGitHub/Marble-Ball-Algorithm/blob/master/Demonstration/img_sceno.png?raw=true)

4. Looked from the top.

      ![](https://github.com/RiverLeeGitHub/Marble-Ball-Algorithm/blob/master/Demonstration/img_top.png?raw=true)

5. Looked from the front.

      ![](https://github.com/RiverLeeGitHub/Marble-Ball-Algorithm/blob/master/Demonstration/img_front.png?raw=true)


## Future Work
Apart from this, this method can also calculate the irregularity of connected regions, or be extended to other multi-dimensional samples and find their main distribution centers to serve data-based decision making.
