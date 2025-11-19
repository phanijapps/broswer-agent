FROM accetto/debian-vnc-xfce-chromium-g3:latest

USER root

# Install Node.js
RUN apt-get update && \
    apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Playwright and @playwright/mcp globally
RUN npm install -g playwright@latest @playwright/mcp@latest

# Install browsers
RUN npx playwright install --with-deps chrome

# Setup screenshots directory
RUN mkdir -p /home/headless/screenshots && \
    chown -R 1000:1000 /home/headless/screenshots

USER 1000

EXPOSE 5901 6901

# Use the base image's default startup - DON'T OVERRIDE
