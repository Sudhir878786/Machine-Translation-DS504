"use strict";

class OnDemandWebService {
    constructor(hostUrl, onReceiveHandler, onErrorHandler) {
        this.dataToSend = null;
        this.inFlight = false;

        this.hostUrl = hostUrl;
        this.onReceiveHandler = onReceiveHandler;
        this.onErrorHandler = onErrorHandler;
    }

    __run__() {
        if (!this.inFlight && this.dataToSend != null) {

            let data = this.dataToSend;

            this.inFlight = true;
            this.dataToSend = null;

            this.__makeRequest__(data)
                .then(data => {
                    this.onReceiveHandler(data.data);
                    this.inFlight = false;
                    this.__run__();
                })
                .catch(error => {
                    this.onErrorHandler(error);
                    this.inFlight = false;
                    this.__run__();
                });
        }
    }

    __makeRequest__(data) {
        throw new Error('You have to implement the method __makeRequest__!');
    }

    sendRequest(dataToSend) {
        console.log(this.inFlight + " | " + JSON.stringify(this.dataToSend) + " | " + this.text);
        this.dataToSend = dataToSend

        this.__run__();
    }

    hasDataToSend() {
        return this.dataToSend != null;
    }
}

class OnDemandTranslator extends OnDemandWebService {

    __makeRequest__(data) {
        console.log("Sending request to API: " + JSON.stringify(data));

        const url = this.hostUrl + "/api/v1/translate";

        return axios.post(url, data)
    }

    translate(sourceText, sourceLang, targetLang) {
        this.sendRequest({
            'source_lang': sourceLang,
            'target_lang': targetLang,
            'input_text': sourceText
        });
    }
}

class OnDemandLanguageDetector extends OnDemandWebService {

    __makeRequest__(data) {
        console.log("Sending request to API: " + JSON.stringify(data));

        const url = this.hostUrl + "/api/v1/predict";

        return axios.post(url, data)
    }

    predictLanguage(text) {
        this.sendRequest({
            'input_text': text
        });
    }
}

class GetStatusWebService {
    constructor(hostUrl) {
        this.hostUrl = hostUrl;
    }

    getStatus() {
        const url = this.hostUrl + "/api/v1/status";
        return axios.get(url);
    }

    waitForStatus(numRepeat) {
        return new Promise((resolve, reject) => {
            var waitForStatusHelper = (numRepeat, prevError) => {
                if (numRepeat > 0) {
                    this.getStatus()
                        .then(() => {
                            resolve();
                        })
                        .catch(error => {
                            waitForStatusHelper(numRepeat - 1, error);
                        });

                } else {
                    reject(prevError);
                }
            };
    
            waitForStatusHelper(numRepeat, null);
        });
    }
}