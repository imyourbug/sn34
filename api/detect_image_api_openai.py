from PIL import Image
import os
import base64
import requests
import time
from dotenv import load_dotenv
import os
from openai import OpenAI


class DetectImageOpenAI:
    def __init__(self) -> None:
        # Load the environment variables from .env file
        load_dotenv()
        self.token = os.getenv("OPENAI_API_KEY")

    # Function to encode the image
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def check_image(self, image_name):
        try:
            # Getting the base64 string
            image_path = f"images/{image_name}"
            base64_image = self.encode_image(image_path)

            # Call model
            client = OpenAI(api_key=self.token)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Is this real image or AI-generated (includes deepfake)? Return 1 if AI is detected and 0 if not. No return other text.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )

            return response.choices[0].message.content
        except Exception as e:
            print(f"Exception call model openAI: {e}")

        return None


input_image = """/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAEAAQADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDwRLaeX/VwyN/uqTVqPRdSl+7Zy/8AAhj+dekOm08cCowea19mdXsI9WcKnhfU3+9HGn+84/pVmPwjcn/WXMS/7oJrsc0mOaOQ0jQpnNxeDocfvLuRv91QKuxeEtNX74lf6v8A4VtKMCpFrPlZ0xpUUr8pmJ4f0yL7tmh/3sn+dWI7G2h5jt40x/dQCrgIprkAcVShdDvCL0RCVxQBS9acik0KBTmNCmnbanSP2qbyc1sqehi52ZSCn0pcVcMAA6Uwx+1S4FqSsVttNYVZK4phXNLlBSRX280YqUrximlaLC0ZER60mOKeRTSKllLQZimkZNSHmm1DWpadhuKeiZ7UgqxGBiqjAiVTUaEFLswKeAM5FDmtFFEuT6EBXnNSIMdaiJO6guapNIyauWtygUwyZqqWPrTg9NzJUe5KSOtMY5FRkkmgdannL9my4W3cUxgBQW4zUDyE8VcHdEVI2ehKBSgioA/HWnB+KUrDptsnWguB0NQmSmk5rG50EhkyaNxxxUWKkUE8VpFmUk2x461YiA4qvipUbBoUtS1G6L4UAUqsOlQLKSKfGcmrUiXAnIBFRMlSMwAxUe6ncXK0iJkINMbC1LJINp5GaoySZ70m0LVDzIBTC2ahLGnBuKhsauxWpmc0HJNJzis2aoMimk0uKbtyaaRMpMVeanHFRotTqtVeyFCN3qIuc0rCpAmKRlqec2dMrsvOaiYVbKcc1EUqXJth7NJEG3NLtFTeWTSGPimiGiLbS4p5Q5GKMY60WEmNZqjOM0pNMJNXF2RlUV2L1o6d6BS49KTlccYaaAMinClAyKUCs7nQkAIqUUwLmnqvrTUh8g4YNOVRmkA5pxocgUCRSAcCrkScZqinWrayYGM04zQ3TdhJWwagklOMCpHOcnNV2x1rW6sYyvexEzEnmomanuaixmpZGlxCaQGg8UHGKzuaWF3UbqTHFKKaYNMXPanIuabjNTRqatWsQk27D40qYLinBcLRt3VlOasddOk+ggFSLGCaRUIqVMg1EJq5pOm0gMKmmfZsmrCZJ5qcAYrRyRnGm2tSgYAvGM1FJH7VoOtVpCK1jG+phOXLoUiuKay1KRk0w+ho5LmbnZFQjNNxinZoPSsOc29mtxv0pyjikpwPtQ2CWo7FOA9KbmnAmpuaWuOXFSBe9MAqYcDpU3NFETbjmg4pSaaaHIpR1HKafu9KiFLk1PU06DmNQuT2qWkZeK1U+5zyp32Kx603vUpWo3wDV89zH2VhmM0m2pAARRt7VDkWqY3GKUdKdjNOCUKQ3BiImasogApiAZqTJPFOUrIdOF2S4yKeg5qJTjjNTwjLVyyZ3wj0Q8pxmlUVLtyuKYVxWKqpM6HSbQ8AClZwB1qEvionfPSt4Svqc9SNtESSSjHWqUk241K3IqnJ1612U6p5lak+hIZPlqAybT1ppyaaVOOa15zkdNsbmj3qLJpd3FcridimiXtThUanipB05qW7G0VceuKcF5qMEA1Kjg8VDNEk9B6inEkUq4xxTtuanmNeS5HmjrTtvNKFockJRY2nAU4LxTglTzmnIxm2nBc1II6djApOY40irIoFVHOSRV2fABqg/JrWm7q5z11Z2HLwKdnnNMUGnbccVTZMUPBxUi8jOKYMVItK5VhATzijfzinNxTAOpovdE2swDkt1q7ASaoKfnq/AcdKzqrQ3w8tS8BlaaynFKjbqm28V5rlys9ZRTRSMfNRMuKvlKgdcVtCq2YzooquvFVZI+avNjFV2XJrppybOOtTRWVOaHj9BUpXBorqi2cNSKWxm7aNlTbfagCp5gVJEQGKeDUmygJzScrlqm1sCLnmpVSlVcVKFrKUjeMEAGKf2puCaeKRQ0KakC09RSniok7s1jGyuxu2jB4p6jNKVOfapuXyiqOKSRgBTtp6UwpmpvqXZ20KMuWbmoTGTVx4+aaIzW6nY5JU7vUqhSOTT1XuamaOl2YFHOCp2ICvPFPAIFSBOc07FPnF7PqR4z1pGwB71KEyacY6amgdNtaFbbjmrMDDimmPNHllTkUTmmgp05Rd0aca8A1ZA4qhbzEDkVcWUEcV5dZSuetSkrCsABVSY5Bx1qdySKhYcU6ejCpqioc55ppIA5qZ15zVOZiGNerTScbo8au5KVmK5GM1Fu9aj3H1pC/FdMLLQ4qjb1Qzk04c0/ApMYNcrZ2WsOUVIEGaYgOasKtZyZtBXGhKmVAaAlSItZtmyiR+XQVxUpBzTXGRVJsmSQ3dik3Zo20vCiqsTckjGanwAOaqiTFL52eKylBs2hUiiZmA6Uwc96YWFIHHaly6Fc6HMlNbhaXfk0khAWhJibVrohJpm7nFShe9MKc1omjJpjiwFJnNJs55p4Wi6BJskQYpWOelNzinIMmob6mqXQdEm481JsBGKbu2jA61KiHrWEpPc2hFbCLCcdKcoZasKpC0hGawdS+5uoJEReq8khzU864HFUHfHWumjFPU5687CvKSMVVkyeaezComeu+Omh5lT3ncjIpjU5344qFmJFdEdThqXTLYHNPVQaXFOA4rjbPRihQoqVBUSnFSBvSoaZrGSRMoycCplQAVHER1NWeoqeRtmntIjNgpCmRU6qMcikJAGK0UbGcpX2KpjxUTLk1bbpURWm5WJ5LlcimZAqZxgcmoGI7UlqDVhpcs2AKlQHHSo1wp5qymCKU3ZFU1djFHNOZNxHoKcV9DTkIA5rCT6o6Ix6Mj2jNBQZpzsueKbnilqOyIyOaXpSM2KYWyatXIbsPLelTxIduahjTJzV1RhayqztobUoX1ZGkWTk1aVaiDgUjTHPFYPmmzdWiWaZIdgzUIn9aUsZPpQqTWrF7RPRDHmGDms+4IJ4q+0QqrNHjNddFxT0OTEKTRQLZNNJpzjmozkV3WPLcrCHpUZpzE9qZVrQxm7s08cU3NSMOKjIrlO9iinr1pgqWMAnFXFXIbsTRAkgdqvqoCVXhQVZzkYFdUIJbnNOo+gxmPemg5NOK0bcCsayS2N8O5PcYVzTSBTzUTA5rkep3IhlXPPaq5AqeTJ4poj9atOyMZK70IkGTU6jApAoHSnH61Mnc0hGwuaR+VxSZxTd9Z2NLjORS5wKOSaMU20KKY0jJpQtGKeBipcilEVPl6VcVgY6qqMmrABwMVjOzN4XQzBZuKf5eKljTHNOYYqHPoVydWVGBD1YjC4zUT4PFRkkdDW0feVjF+47ouNt9RVScKQai3nPWl3AjrW0aXLqYzq82hUdPmqFk4qy5qI11wZ59VFYpUbLg1bI9qhdec1oYNGgwqEg1ZYGoytc0TukRqKlRcmgCnqOauJm1oWYeOKsqpqCADHPWpw4UHNdUF3OaemwpIHWo3kFV5ZiTxUJkNZVFc1pSa3LW+mFt3FRB+OtKHHrXJJWO6EroGAFRlsGnFs9KYTSQ2+wFsVGzkGgtTGxTRLY7eWNOFRdKeDUSLiSjGKBTaC9Yu5urIGbFOTmmKmeTViNKTaSHFNscoqwmMdKRIxUoCrXO5HQkKDgdKieUc+tPaZADzzVQ5ZiaqFNy1ZE6iWiBjuNOVdy0qx8dKkC4FbrTYxepTkTB4qAlhnNXXXrUJizXZCV1qcNSFndFbk04JmrKWxPIFSiDA6Vuloc77MoFKidMVovF7VXlj44pXDkuTnBNMIBqQgYqPisrWN73BF5xUoUCmp1p78U1pqG+ghfaeKRpCaaTml25qlUIdO7I25NMzUrVC/tRzXDksJvoDZqM9aUdaTQJkoJpGoB4pSCaye5vHVERNNx61KVphpXuK1hMc1Iq5pqLzVuNOORWc3Y2pq5EENKI/arJUUojzWDkdKiQqnFSgbaeFwOlMzk1F7lbDgxpCpbqaVVOamCjFS3bYq19yrsOakRamKcUirWildGbjZiqtOKZFOUYpykU1uKRWMfNHlgGrewEVEyZbFdcGjlnF3CLaOKeyA9KaqbW6VI5wPeuiM1bU5pU3e6Kk2FqlIQatS5JOaqOuancHdIcXOKjzStUfekgehYQ8UrMMVErcUM2aGCY4mjzcDFRFqYTmkkU5diVn4qItk0hPam/jVaEXYvWlHWkBpRTF1JV60/HFRqMmpscVzz3OqGxC1MK5qwI8mjy8UlJIHFsbEnNW1HFMRMCp0TNY1Hc3pxtoASpAvFGMCmmTtXO9djpWgNTFXnNP60qrSvZDsKF4ppbBxUhBxgUzZg5IqU11G0PU5GKUrSJxSyPtFCeugW0GFwOKFcVAzZJpufSulI52XBLil84A1RLH1pVbB5NaqyRm7tl8SgjmmySgjA61W34HWoWkxzmiEtdRTjoPkbrUHFNecmod5zmuxNHC73Hk5pMUc5oqR6iU0mlam9aaYmhCaM0EUlJsLCHJpKd3pMc0rjsKBT1FNAqZFpuQRg2x0a81ZVM1Go4qeMgda46knc7qUUlYa67RTVG5qkcbjQq4NSnZFtXY9UGKlUDFMB4qRRxWM3oaxI3FQ4+apmzmmheaE7IGrsVamUc9KjUVMuAKylc1Q7bSMKXcKTdxWfK7juiM8VHIc0pY5pDzW8Y2M5STIduOaacip8cU1lrdMxZWNJup7AVEwrVWM2PLVC7c04mmYppEt3GE0ypSKTbWikYuA/HNBFOZaaBQncGrEZFNzUrCo9vNUmQ1qNpCKdjmlxQKwzBpwFO21Iic0mylEaqVMoFOVacFFZt3NkrDlXilUc0oOBSqOayfmbIkVMmnbOKVTilLYrncjdJIj6VIGwKiY804cihrQaYp5pAKWkzzSSbBuxIopXNItKea6KdG7uzGpVtsMXJqTGBzQpQd6R5QBVypozVTQYQMmmkgCoy5JyKYXpKA+clzmms3amhieBT1TjmlyhzEJUmmMtWGFQORWkYMzlNEJFJUhBNJtzWnKZ8zGUgGTUoSnCPmpZa1HsnrUZAAqdzUDc1EW2VJIjIzzTCOakpQtaJmTRDtpwWpNtO20Ngo6jAvFOXIp4WlC81k5G0YgtPxQF5p4Ws3M1UBoFP6UvSkJzWd3IuyihC9AcmkIpBT0Fdj+tOHSmCnina4r2AU0/eqTAFRnk1rCJnOROCAtQySmjPFRkZNaJ20M2rjC5z1o3k9aQrzQBincVh4PFMPJp1KF5oAegwKkJ4oVTigrQldjbsiEnJphjJqRkwacK20sYWbZEI6cEqUilRc9awm7I2hG5GI6kVOKlC07bXNKZ0xgZ8gNR4xVgqT1ppSuhbGDIAOaeFp23NOC0cwlEbtpQoqQLSheKhyLUSMinKtSbKcFwKxlI1hEYFpwFO280uDWerNdiIijbUu2lxxWiiZNkODSbeam2ikxVcrFcZigVJtzSbfarjFkOQw5ppFS7abitVoQ9RuPagr7U/IFBNDjcSZGUyKZt5qftSbcnNOwXIwtOwBTuAaaOWpqBLmh27C8daRSWNOAyalRVHNaxikZSm2QshNAiOOnFWWApRg0pDgmV9mKfjApzkLUTSVyTudcLDt2DThIKrM+ajEoHU1Hs2yvaJH/9k=
"""
path = "images"
os.makedirs(path, exist_ok=True)
img_name = f"output_image_{int(time.time())}.jpeg"
with open(f"images/{img_name}", "wb") as image_file:
    image_bytes = base64.b64decode(input_image)
    image_file.write(image_bytes)

detectImageAPI = DetectImageOpenAI()
result = detectImageAPI.check_image(img_name)
print(f"Run detectImageAPI {result}")
