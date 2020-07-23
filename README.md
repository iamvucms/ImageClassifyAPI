# IMAGE CLASSIFY SERVER 

This a simple image classify server use Flask + CNN,  with 7 classes:
- Fashion
- Cosplay
- Art
- Architecture
- Landscape
- Decor
- Food

This server serves for project [React Native Instagram Clone](https://github.com/iamvucms/react-native-instagram-clone)
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install.

```bash
git clone https://github.com/iamvucms/ImageClassifyAPI.git
cd ImageClassifyAPI
pip3 install requirements.txt
flask run --host=0.0.0.0 --port=YOUR_PORT
```

## Usage

Allow POST Method to: [http://0.0.0.0:YOUR_PORT](http://0.0.0.0:YOUR_PORT)

Example Typescript React
```
const CLASSIFY_API = `http://0.0.0.0:YOUR_PORT/classify`
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
