$(function() {
    console.log('jq is working');
    createVis();
});


function createVis() {
    var width = 960,
    height = 500,
    padding = 1.5,
    clusterPadding = 6,
    maxRadius = 12;

    var num_clusters = 4;

    var color = d3.scale.category10().domain(d3.range(num_clusters));
    //var color = d3.scaleOrdinal(d3.schemeCategory20);
    var c = new Array(num_clusters);

    var svg = d3.select("body").append("svg").attr("width", width)
        .attr("height", height);

   /* var makeClusters =  function makeClusters(c) {
        d1 = {cluster: 1, radius: 1};
        d2 = {cluster: 2, radius: 1};
        d3 = {cluster: 3, radius: 1};
        d4 = {cluster: 4, radius: 1};
        c[0] = d1;
        c[1] = d2;
        c[2] = d3;
        c[3] = d4;
        return c;
    };

    */

      var d1 = {cluster: 1, rating: 1};
        var d2 = {cluster: 2, rating: 1};
        var da3 = {cluster: 3, rating: 1};
        var d4 = {cluster: 4, rating: 1};
        c[0] = d1;
        c[1] = d2;
        c[2] = da3;
        c[3] = d4;

    clusters = c;

    var lay = d3.layout.pack();

    d3.json('/data2', function (error, graph) {
        if (error) throw error;

    //d3.layout.pack()
        lay
        .sort(null).size([width, height]).children(function(d) { return d.values; })
            .value(function(d) { return d.rating; })
            .nodes({values: d3.nest()
                .key(function(d) {return d.cluster; })
                .entries(graph.nodes)});
    
       console.log(lay.nodes[0]);

        var force = d3.layout.force()
            .nodes(graph.nodes)
            .size([width, height])
            .gravity(.02)
            .charge(0)
            .on("tick", tick)
            .start();

        var node = svg.selectAll("circle")
            .data(graph.nodes)
            .enter().append("circle")
            .style("fill", function(d) { return color(d.cluster); })
            .call(force.drag);

        node.transition()
            .duration(750)
            .delay(function(d, i)  {return i*5; })
            .attrTween("r", function(d) {
                var i = d3.interpolate(0, d.rating);
                return function(t) {return d.rating = i(t); };
            });

   // });

    function tick(e) {
        //console.log("tick e");
        //console.log(e);
        node.each(cluster(10 * e.alpha * e.alpha))
            .each(collide(.5))
            .attr("cx", function(d) { return d.x; })
            .attr("cy", function(d) { return d.y; });
    };

     function cluster(alpha) {
       return function(d) {
          //console.log("cluster d");
          //console.log(d)
          var cluster = clusters[d.cluster];
          //console.log(cluster)
          if (cluster === d) return;
          var x = d.x - cluster.x,
          y = d.y - cluster.y,
          l = Math.sqrt(x * x + y * y),
          r = d.rating + cluster.rating;
          if (l != r) {
            l = (l - r) / l * alpha;
            d.x -= x *= l;
            d.y -= y *= l;
            cluster.x += x;
            cluster.y += y;
          };
        }; 
    }

    function collide(alpha) {
        var quadtree = d3.geom.quadtree(graph.nodes);
        return function (d) {
            //console.log(d.rating)
            var r = d.rating + maxRadius + Math.max(padding, clusterPadding),
                nx1 = d.x - r,
                nx2 = d.x + r,
                ny1 = d.y - r,
                ny2 = d.y + r;
                //console.log(d.x);
            quadtree.visit(function(quad, x1, y1, x2, y2) {
                //console.log(quad.point);
                //console.log(quad);
                //if (!quad.point) console.log("h");
                if (quad.point && (quad.point !== d)) {
                    var x = d.x - quad.point.x,
                        y = d.y - quad.point.y,
                        l = Math.sqrt(x*x + y*y),
                        r = d.rating + quad.point.rating + (d.cluster === quad.point.cluster ? padding : clusterPadding);
                    console.log(r);
                    if (l < r) {
                        l = (l - r) / l * alpha;
                        d.x -= x; // *= l;
                        d.y -= y; // *= l;
                        quad.point.x += x;
                        quad.point.y += y;
                        console.log(quad.point.x);
                    };
                };
                var o = x1 > nx2 || x2 < nx1 || y1 > ny2 || y2 < ny1;
                //console.log(o)
                return x1 > nx2 || x2 < nx1 || y1 > ny2 || y2 < ny1;
            });
        };
    }; 
  });

   
} //closes createVis()
