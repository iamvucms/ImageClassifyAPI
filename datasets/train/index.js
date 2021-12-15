const fs = require("fs");
const request = require("request");
const download = function (uri, filename, callback) {
  request.head(uri, function (err, res, body) {
    request(uri).pipe(fs.createWriteStream(filename)).on("close", callback);
  });
};
const PAGE = 1;
const TYPE = "dance";
request(
  `https://api.500px.com/v1/photos/search?type=photos&term=${TYPE}&image_size%5B%5D=1&image_size%5B%5D=2&image_size%5B%5D=32&image_size%5B%5D=31&image_size%5B%5D=33&image_size%5B%5D=34&image_size%5B%5D=35&image_size%5B%5D=36&image_size%5B%5D=2048&image_size%5B%5D=4&image_size%5B%5D=14&include_states=true&formats=jpeg%2Clytro&include_tags=true&exclude_nude=true&page=${PAGE}&rpp=200`,
  {},
  (err, res, body) => {
    const json = JSON.parse(body);
    const photos = json.photos;
    photos.map((photo, index) => {
      const idx = (PAGE - 1) * 200 + index;
      console.log("downloading----" + idx);
      const imageUrl = photo.image_url.pop();
      download(imageUrl, `${TYPE}/img_${idx}.jpeg`, () => {});
    });
  }
);
