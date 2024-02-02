    package com.example.customizaesdeimagens.activity.model;

    import android.app.Activity;
    import android.content.Intent;
    import android.graphics.Bitmap;
    import android.os.AsyncTask;
    import android.provider.MediaStore;
    import android.util.Base64;
    import android.util.Log;

    import java.io.ByteArrayOutputStream;
    import java.io.DataOutputStream;
    import java.io.InputStream;
    import java.net.HttpURLConnection;
    import java.net.URL;

    public class Funcoes {
        private String tituloFuncao;
        private int position;

        public static final int PICK_IMAGE_REQUEST = 1;

        public Funcoes() {
        }

        public Funcoes(String tituloFuncao, int position) {
            this.tituloFuncao = tituloFuncao;
            this.position = position;
        }

        public String getTituloFuncao() {
            return tituloFuncao;
        }

        public void setTituloFuncao(String tituloFuncao) {
            this.tituloFuncao = tituloFuncao;
        }

        public int getPosition() {
            return position;
        }

        public void setPosition(int position) {
            this.position = position;
        }

        public void getFuncoesParametro(Activity activity) {
            Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            intent.setType("image/*");
            activity.startActivityForResult(intent, PICK_IMAGE_REQUEST);
        }

        public void processarImagemSelecionada(Bitmap imagemBitmap) {
            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
            imagemBitmap.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
            byte[] byteArray = byteArrayOutputStream.toByteArray();
            String encodedImage = Base64.encodeToString(byteArray, Base64.DEFAULT);

            Log.d("Posição antes do envio:", String.valueOf(this.position));

            // Enviar a imagem para o servidor Python
            new EnviarImagemParaServidor(String.valueOf(this.position)).execute(encodedImage, String.valueOf(this.position));

        }

        private static class EnviarImagemParaServidor extends AsyncTask<String, Void, String> {
            private int position;
            public EnviarImagemParaServidor(String position) {
                this.position = Integer.parseInt(position);
            }

            @Override
            protected String doInBackground(String... params) {
                try {
                    String imagemBase64 = params[0];
                    String position = params[1];  // Obter a posição do segundo parâmetro

                    URL url = new URL("http://10.0.2.2:5000/process_image");
                    HttpURLConnection connection = (HttpURLConnection) url.openConnection();
                    connection.setRequestMethod("POST");
                    connection.setRequestProperty("Content-Type", "multipart/form-data;boundary=*");
                    connection.setDoOutput(true);

                    DataOutputStream outputStream = new DataOutputStream(connection.getOutputStream());
                    String boundary = "*";
                    String lineEnd = "\r\n";

                    outputStream.writeBytes("--" + boundary + lineEnd);
                    outputStream.writeBytes("Content-Disposition: form-data; name=\"image\"; filename=\"image.jpg\"" + lineEnd);
                    outputStream.writeBytes("Content-Type: image/jpeg" + lineEnd);
                    outputStream.writeBytes(lineEnd);
                    outputStream.write(Base64.decode(imagemBase64, Base64.DEFAULT));
                    outputStream.writeBytes(lineEnd);
                    outputStream.writeBytes("--" + boundary + lineEnd);

                    // Adicionar a posição ao corpo da solicitação
                    outputStream.writeBytes("Content-Disposition: form-data; name=\"position\"" + lineEnd);
                    outputStream.writeBytes(lineEnd);
                    outputStream.writeBytes(position);
                    outputStream.writeBytes(lineEnd);
                    outputStream.writeBytes("--" + boundary + "--" + lineEnd);

                    outputStream.flush();
                    outputStream.close();

                    int responseCode = connection.getResponseCode();
                    if (responseCode == HttpURLConnection.HTTP_OK) {
                        InputStream in = connection.getInputStream();
                        // Processar a resposta conforme necessário
                        // ...

                        return "Sucesso";
                    } else {
                        return "Erro: " + responseCode;
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                    return "Erro: " + e.getMessage();
                }
            }

            @Override
            protected void onPostExecute(String result) {
                Log.d("Resultado do servidor", result);
            }
        }
    }