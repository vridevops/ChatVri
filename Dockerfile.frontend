FROM nginx:alpine

# Copiar HTML al directorio de nginx
COPY index.html /usr/share/nginx/html/

# Exponer puerto 80
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]