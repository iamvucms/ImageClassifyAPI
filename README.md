# IMAGE CLASSIFY SERVER 

This a simple image classify server use Flask + CNN,  with 7 classes:
- Fashion
- Cosplay
- Art
- Architecture
- Landscape
- Decor
- Food

Accuracy: ~80%

This server serves for project [React Native Instagram Clone](https://github.com/iamvucms/react-native-instagram-clone)
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install.

```bash
git clone https://github.com/iamvucms/ImageClassifyAPI.git
cd ImageClassifyAPI
pip3 install -r requirements.txt
flask run --host=0.0.0.0 --port=YOUR_PORT
OR
sudo chmod 755 run.sh
./run.sh
```

## Usage

Allow POST Method to: [http://0.0.0.0:YOUR_PORT](http://0.0.0.0:YOUR_PORT)

Example Typescript React
```javascript
const CLASSIFY_API = `http://0.0.0.0:YOUR_PORT/classify`//<-- change to 0.0.0.0 your private IP(ex:192.168.1.5) if use it for Mobile App (use ifconfig command to get)
export const getImageClass = (imageUrl: string): Promise<string> => {
    return new Promise((resolve, reject) => {
        const data = new FormData()
        data.append('URL', imageUrl)
        fetch(CLASSIFY_API, {
            method: 'POST',
            body: data
        }).then(res => res.json())
            .then(result => {
                if (result.success) {
                    resolve(result.class_name)
                } else reject('Error')
            })
    })
} 
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
