import request from "request";
import fs from 'fs';

// 腾讯地图获取区域文档
// https://lbs.qq.com/service/webService/webServiceGuide/webServiceDistrict
// 腾讯地图key
const key = 'LNABZ-UXM6G-SDNQK-IE6KK-T44VE-SCFUE';

// 获取一级省份列表
async function getProvince() {
    return new Promise((resolve, reject) => {
        request({
            method: 'get',
            uri: `https://apis.map.qq.com/ws/district/v1/getchildren?key=${key}`,
            json: true,
        }, (err, res, body) =>{
            if (body.status == 0) {
                resolve(body.result[0]);
            } else {
                console.log('error');
            }
        });
    });
}

// 获取子级数据列表
async function getChildren(id) {
    return new Promise((resolve, reject) => {
        request({
            method: 'get',
            uri: `https://apis.map.qq.com/ws/district/v1/getchildren?key=${key}&id=${id}`,
            json: true,
        }, (err, res, body) =>{
            if (body.status == 0) {
                // 并发限制：5次/秒/接口/Key
                setTimeout(()=>{
                    resolve(body.result[0]);
                },250);
            } else {
                console.log('error');
            }
        });
    });
}

async function getCode() {
    console.log('生成中,请等待');
    let res = await getProvince();
    // 一层省份ID  北京|天津|上海|重庆|香港|澳门
    let array = ['110000','120000','310000','500000','810000','820000'];
    let codeData = [];
    for(let pEle of res) {
        let province = {
            pid: pEle.id,
            name: pEle.name,
            fullname: pEle.fullname,
            children: [],
        };
        // 获取省份城市数据列表
        if (array.indexOf(pEle.id) == -1) {
            let cityRes = await getChildren(pEle.id);
            for (let cEle of cityRes) {
                let city = {
                    cid: cEle.id,
                    name: cEle.name || cEle.fullname,
                    fullname: cEle.fullname,
                    children: [],
                };
                province.children.push(city);
            }
        } else {
            let city = {
                cid: pEle.id,
                name: pEle.name || pEle.fullname,
                fullname: pEle.fullname,
                children: [],
            }
            province.children.push(city);
        }
        codeData.push(province);
    }
    // 获取区域详情
    for (let pEle of codeData) {
        for (let cEle of pEle.children) {
            let areaRes = await getChildren(cEle.cid);
            areaRes.forEach(aele => {
                let area = {
                    aid: aele.id,
                    name: aele.name || aele.fullname,
                    fullname: aele.fullname,
                };
                cEle.children.push(area);
            });
        }
    }
    fs.writeFile('./data.json',JSON.stringify(codeData), err => {
        if (err) {
            console.error(err);
            return;
        }
        console.log('写入成功');
    });

}
getCode();
